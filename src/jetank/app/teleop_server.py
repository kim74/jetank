"""
Remote-control web server: serves the control page and receives commands over a WebSocket.

Layer: app

``GET /`` returns the page; the page opens a WebSocket on ``/ws`` and exchanges the JSON
messages defined in :mod:`jetank.domain.teleop`. All decisions about who may drive live in
:class:`~jetank.control.teleop.TeleopSession`; this module only moves bytes and runs the
watchdog timer.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from importlib import resources
from typing import TYPE_CHECKING

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from jetank.domain.teleop import parse_client_message
from jetank.errors import HardwareError, ProtocolError, ServerError

if TYPE_CHECKING:
    from collections.abc import Callable

    from jetank.config.settings import TeleopConfig
    from jetank.control.teleop import TeleopSession

logger = logging.getLogger(__name__)

PAGE_PATHS = frozenset({"/", "/index.html"})
WEBSOCKET_PATH = "/ws"
# How long shutdown waits for a client to acknowledge closing (the library default is 10 s).
CLOSE_TIMEOUT_S = 1.0


def load_page() -> bytes:
    """The control page shipped with the package."""
    return resources.files("jetank.app").joinpath("static/teleop.html").read_bytes()


class TeleopServer:
    """Serves one :class:`TeleopSession` over HTTP + WebSocket until stopped.

    Args:
        session: Arbitration and drive logic.
        config: Listen address and watchdog tick period.
        page: HTML returned for ``GET /``.
    """

    def __init__(self, session: TeleopSession, config: TeleopConfig, page: bytes) -> None:
        self._session = session
        self._config = config
        self._page = page
        self._client_ids = itertools.count(1)
        self._stop: asyncio.Event | None = None
        self._failure: HardwareError | None = None

    async def serve(self, on_ready: Callable[[int], None] | None = None) -> None:
        """Run until :meth:`request_stop` is called or the motors fail.

        Args:
            on_ready: Called with the bound TCP port once the server accepts connections.

        Raises:
            ServerError: The address cannot be bound (e.g. the port is in use).
            HardwareError: The motors failed; the robot has been stopped as far as possible.
        """
        self._stop = asyncio.Event()
        try:
            server = await serve(
                self._handle_client,
                self._config.host,
                self._config.port,
                process_request=self._process_request,
                close_timeout=CLOSE_TIMEOUT_S,
            )
        except OSError as exc:
            address = f"{self._config.host}:{self._config.port}"
            raise ServerError(f"cannot listen on {address}: {exc.strerror}") from exc
        async with server:
            port = int(next(iter(server.sockets)).getsockname()[1])
            logger.info("teleop server listening on %s:%d", self._config.host, port)
            if on_ready is not None:
                on_ready(port)
            watchdog = asyncio.create_task(self._run_watchdog())
            try:
                await self._stop.wait()
            finally:
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog
        if self._failure is not None:
            raise self._failure

    def request_stop(self) -> None:
        """Ask :meth:`serve` to return. Safe to call before the server has started."""
        if self._stop is not None:
            self._stop.set()

    def _fail(self, exc: HardwareError) -> None:
        logger.error("teleop: motor failure, shutting down: %s", exc)
        self._failure = exc
        self.request_stop()

    async def _run_watchdog(self) -> None:
        while True:
            try:
                self._session.tick()
            except HardwareError as exc:
                self._fail(exc)
                return
            await asyncio.sleep(self._config.tick_period_s)

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if request.path == WEBSOCKET_PATH:
            return None  # continue with the WebSocket handshake
        if request.path in PAGE_PATHS:
            headers = Headers(
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(self._page))),
                    ("Cache-Control", "no-store"),
                ]
            )
            return Response(200, "OK", headers, self._page)
        return connection.respond(404, "Not found\n")

    async def _handle_client(self, connection: ServerConnection) -> None:
        client_id = f"client-{next(self._client_ids)}@{connection.remote_address}"
        try:
            await connection.send(self._session.connect(client_id).model_dump_json())
            async for raw in connection:
                try:
                    message = parse_client_message(raw)
                except ProtocolError:
                    logger.warning("teleop: invalid message from %s: %.200r", client_id, raw)
                    status = self._session.reject(client_id, "invalid message")
                else:
                    status = self._session.handle(client_id, message)
                await connection.send(status.model_dump_json())
        except ConnectionClosed:
            pass
        except HardwareError as exc:
            self._fail(exc)
        finally:
            try:
                self._session.disconnect(client_id)
            except HardwareError as exc:
                self._fail(exc)
