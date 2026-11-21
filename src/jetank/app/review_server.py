"""
Review pages for saved records: the small HTTP server shared by the pose and scene reviews.

Layer: app

A review is a tool of its own, not part of the robot control app: it opens no servo bus,
camera or motors, only a records directory. It can run while teleop is stopped, or on a laptop
with a copy of the directory. Like teleop there is no authentication, so only run it on a
trusted network; it can change and delete records (deleted ones go to ``trash/``), never
anything else.

``GET /`` returns the page. The page talks to a small JSON API:

- ``GET /api/records``: the listing (what is in it depends on the review)
- ``GET /api/records/<id>/image``: the JPEG saved with a record
- ``PUT /api/records/<id>``: new labels for a record (a JSON body)
- ``DELETE /api/records/<id>``: delete a record
"""

from __future__ import annotations

import logging
import re
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar
from urllib.parse import urlsplit

from pydantic import BaseModel, ValidationError

from jetank.app.signals import raise_on_termination_signals
from jetank.errors import JetankError, RecordNotFoundError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

#: Largest request body accepted; a label edit is a few hundred bytes.
MAX_BODY_BYTES: Final = 4096
PAGE_PATHS: Final = frozenset({"/", "/index.html"})
RECORDS_PATH: Final = "/api/records"
_RECORD_PATH: Final = re.compile(r"^/api/records/([^/]+)(/image)?$")
_JSON: Final = "application/json"

L = TypeVar("L", bound=BaseModel)
L_contra = TypeVar("L_contra", bound=BaseModel, contravariant=True)


class ReviewableStore(Protocol[L_contra]):
    """What a review needs from a record store (the grasp and scene stores both fit)."""

    def load_image(self, record_id: str) -> bytes | None: ...

    def relabel(self, record_id: str, labels: L_contra) -> BaseModel: ...

    def delete(self, record_id: str) -> None: ...


@dataclass(frozen=True)
class Reply:
    """An HTTP response, before it is written to the socket."""

    status: HTTPStatus
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: dict[str, str] = field(default_factory=lambda: {"Cache-Control": "no-store"})


def load_page(name: str) -> bytes:
    """A page shipped with the package, e.g. ``"pose_review.html"``."""
    return resources.files("jetank.app").joinpath(f"static/{name}").read_bytes()


class RecordReviewApi(ABC, Generic[L]):
    """Answers a review page's requests from a record store; knows nothing about sockets.

    Subclasses say what the listing holds.

    Args:
        store: Where the records are.
        labels_model: What a ``PUT`` body must parse as.
        page: HTML returned for ``GET /``.
        kind: What a record is, for messages (e.g. "scene").
    """

    def __init__(
        self, store: ReviewableStore[L], labels_model: type[L], page: bytes, kind: str
    ) -> None:
        self._reviewable = store
        self._labels_model = labels_model
        self._page = page
        self._kind = kind

    @abstractmethod
    def listing(self) -> BaseModel:
        """What ``GET /api/records`` returns.

        Raises:
            StorageError: The records cannot be read.
        """

    def handle(self, method: str, target: str, body: bytes, content_type: str | None) -> Reply:
        """Answer one request.

        Args:
            method: HTTP method, e.g. ``"GET"``.
            target: Request target, e.g. ``"/api/records?x=1"`` (the query is ignored).
            body: Request body (empty when there is none).
            content_type: The request's ``Content-Type`` header, if any.
        """
        try:
            return self._route(method, urlsplit(target).path, body, content_type)
        except RecordNotFoundError as exc:
            return _text(HTTPStatus.NOT_FOUND, str(exc))
        except StorageError as exc:
            logger.error("%s review: %s", self._kind, exc)
            return _text(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _route(self, method: str, path: str, body: bytes, content_type: str | None) -> Reply:
        if path in PAGE_PATHS or path == RECORDS_PATH:
            if method != "GET":
                return _not_allowed(method)
            if path == RECORDS_PATH:
                return Reply(HTTPStatus.OK, self.listing().model_dump_json().encode(), _JSON)
            return Reply(HTTPStatus.OK, self._page, "text/html; charset=utf-8")
        match = _RECORD_PATH.match(path)
        if match is None:
            return _text(HTTPStatus.NOT_FOUND, "not found")
        record_id, image = match.groups()
        if image is not None:
            return self._image(record_id) if method == "GET" else _not_allowed(method)
        if method == "PUT":
            return self._relabel(record_id, body, content_type)
        if method == "DELETE":
            self._reviewable.delete(record_id)
            return Reply(HTTPStatus.NO_CONTENT)
        return _not_allowed(method)

    def _image(self, record_id: str) -> Reply:
        jpeg = self._reviewable.load_image(record_id)
        if jpeg is None:
            return _text(HTTPStatus.NOT_FOUND, f"{self._kind} {record_id} has no image")
        # Images never change once saved, so the browser may keep them.
        return Reply(HTTPStatus.OK, jpeg, "image/jpeg", {"Cache-Control": "private, max-age=3600"})

    def _relabel(self, record_id: str, body: bytes, content_type: str | None) -> Reply:
        # Requiring JSON also means a page from another site cannot send this without a CORS
        # preflight, which this server never grants.
        if content_type is None or content_type.split(";")[0].strip() != _JSON:
            return _text(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"labels must be sent as {_JSON}")
        try:
            labels = self._labels_model.model_validate_json(body)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
                for error in exc.errors()
            )
            return _text(HTTPStatus.BAD_REQUEST, f"invalid labels: {problems}")
        record = self._reviewable.relabel(record_id, labels)
        return Reply(HTTPStatus.OK, record.model_dump_json().encode(), _JSON)


def _text(status: HTTPStatus, message: str) -> Reply:
    return Reply(status, message.encode())


def _not_allowed(method: str) -> Reply:
    return _text(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} is not allowed here")


def make_server(api: RecordReviewApi[L], host: str, port: int) -> HTTPServer:
    """An HTTP server that passes every request to ``api``, one at a time.

    Raises:
        OSError: The address cannot be bound (e.g. the port is in use).
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "jetank-review"

        def do_GET(self) -> None:
            self._dispatch()

        def do_PUT(self) -> None:
            self._dispatch()

        def do_DELETE(self) -> None:
            self._dispatch()

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("review: %s " + format, self.address_string(), *args)

        def _dispatch(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY_BYTES:
                self._send(_text(HTTPStatus.BAD_REQUEST, "missing or oversized request body"))
                return
            body = self.rfile.read(length) if length else b""
            self._send(api.handle(self.command, self.path, body, self.headers.get("Content-Type")))

        def _send(self, reply: Reply) -> None:
            self.send_response(reply.status)
            if reply.status is not HTTPStatus.NO_CONTENT:
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(reply.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.end_headers()
            if reply.status is not HTTPStatus.NO_CONTENT:
                self.wfile.write(reply.body)

    return HTTPServer((host, port), Handler)


def serve(api: RecordReviewApi[L], host: str, port: int, title: str, directory: Path) -> int:
    """Serve a review page until Ctrl-C or SIGTERM.

    Args:
        api: The review.
        host: Interface to listen on.
        port: TCP port (0 picks a free one).
        title: What is printed, e.g. "pose review".
        directory: The records directory, for the printed line.

    Returns:
        The exit code, 0.

    Raises:
        JetankError: The port cannot be opened.
    """
    try:
        server = make_server(api, host, port)
    except OSError as exc:
        raise JetankError(f"could not listen on {host}:{port}: {exc}") from exc
    with server, raise_on_termination_signals():
        print(
            f"{title} at http://{socket.gethostname()}.local:{server.server_port}/ "
            f"for the records in {directory} (Ctrl-C stops)"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt as exc:
            logger.info("%s stopped: %s", title, exc or "Ctrl-C")
    return 0
