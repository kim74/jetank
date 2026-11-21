"""
Pose review: a web page for looking through taught grasp records, fixing labels, deleting bad ones.

Layer: app

This is a tool of its own, not part of the robot control app: it opens no servo bus, camera or
motors, only the records directory. It can run while teleop is stopped, or on a laptop with a
copy of the directory. Like teleop there is no authentication, so only run it on a trusted
network; it can change and delete records (deleted ones go to ``trash/``, see
:class:`~jetank.drivers.grasp_directory.GraspDirectory`), never anything else.

``GET /`` returns the page. The page talks to a small JSON API:

- ``GET /api/records``: every record (newest first), the fit summary and suggested poses
- ``GET /api/records/<id>/image``: the JPEG saved with a record
- ``PUT /api/records/<id>``: new labels for a record (a :class:`TeachLabels` JSON body)
- ``DELETE /api/records/<id>``: delete a record
"""

from __future__ import annotations

import logging
import re
import socket
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import resources
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from jetank.app.arm_commands import format_fit, format_poses
from jetank.app.signals import raise_on_termination_signals
from jetank.domain.teaching import (
    GraspRecord,
    GraspResult,
    GripPoint,
    TeachLabels,
    TeachStage,
    ToyPosture,
    summarize_fit,
)
from jetank.drivers.grasp_directory import GraspDirectory
from jetank.errors import JetankError, RecordNotFoundError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.interfaces.grasp_store import GraspStore

logger = logging.getLogger(__name__)

DEFAULT_PORT: Final = 8081
#: Largest request body accepted; a label edit is a few hundred bytes.
MAX_BODY_BYTES: Final = 4096
PAGE_PATHS: Final = frozenset({"/", "/index.html"})
RECORDS_PATH: Final = "/api/records"
_RECORD_PATH: Final = re.compile(r"^/api/records/([^/]+)(/image)?$")
_JSON: Final = "application/json"


@dataclass(frozen=True)
class Reply:
    """An HTTP response, before it is written to the socket."""

    status: HTTPStatus
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: dict[str, str] = field(default_factory=lambda: {"Cache-Control": "no-store"})


class ReviewListing(BaseModel):
    """What the page shows: every record, the summary and the label choices.

    Attributes:
        records: All records, newest first.
        summary: The same text ``jetank poses`` prints (fit table and config snippet).
        choices: Allowed values of each label field, so the page's menus match the domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[GraspRecord]
    summary: list[str]
    choices: dict[str, list[str]]


_CHOICES: Final = {
    "stage": [s.value for s in TeachStage],
    "posture": [p.value for p in ToyPosture],
    "grip_point": [g.value for g in GripPoint],
    "result": [r.value for r in GraspResult],
}


def load_page() -> bytes:
    """The review page shipped with the package."""
    return resources.files("jetank.app").joinpath("static/pose_review.html").read_bytes()


class PoseReviewApi:
    """Answers the page's requests from a :class:`GraspStore`; knows nothing about sockets.

    Args:
        store: Where the records are.
        page: HTML returned for ``GET /``.
    """

    def __init__(self, store: GraspStore, page: bytes) -> None:
        self._store = store
        self._page = page

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
            logger.error("pose review: %s", exc)
            return _text(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _route(self, method: str, path: str, body: bytes, content_type: str | None) -> Reply:
        if path in PAGE_PATHS or path == RECORDS_PATH:
            if method != "GET":
                return _text(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} is not allowed here")
            if path == RECORDS_PATH:
                return self._listing()
            return Reply(HTTPStatus.OK, self._page, "text/html; charset=utf-8")
        match = _RECORD_PATH.match(path)
        if match is None:
            return _text(HTTPStatus.NOT_FOUND, "not found")
        record_id, image = match.groups()
        if image is not None:
            if method != "GET":
                return _text(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} is not allowed here")
            return self._image(record_id)
        if method == "PUT":
            return self._relabel(record_id, body, content_type)
        if method == "DELETE":
            self._store.delete(record_id)
            return Reply(HTTPStatus.NO_CONTENT)
        return _text(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} is not allowed here")

    def _listing(self) -> Reply:
        records = self._store.load_all()
        summary = [*format_fit(summarize_fit(records)), "", *format_poses(records)]
        listing = ReviewListing(records=records[::-1], summary=summary, choices=_CHOICES)
        return Reply(HTTPStatus.OK, listing.model_dump_json().encode(), _JSON)

    def _image(self, record_id: str) -> Reply:
        jpeg = self._store.load_image(record_id)
        if jpeg is None:
            return _text(HTTPStatus.NOT_FOUND, f"grasp record {record_id} has no image")
        # Images never change once saved, so the browser may keep them.
        return Reply(HTTPStatus.OK, jpeg, "image/jpeg", {"Cache-Control": "private, max-age=3600"})

    def _relabel(self, record_id: str, body: bytes, content_type: str | None) -> Reply:
        # Requiring JSON also means a page from another site cannot send this without a CORS
        # preflight, which this server never grants.
        if content_type is None or content_type.split(";")[0].strip() != _JSON:
            return _text(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"labels must be sent as {_JSON}")
        try:
            labels = TeachLabels.model_validate_json(body)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
                for error in exc.errors()
            )
            return _text(HTTPStatus.BAD_REQUEST, f"invalid labels: {problems}")
        record = self._store.relabel(record_id, labels)
        return Reply(HTTPStatus.OK, record.model_dump_json().encode(), _JSON)


def _text(status: HTTPStatus, message: str) -> Reply:
    return Reply(status, message.encode())


def make_server(api: PoseReviewApi, host: str, port: int) -> HTTPServer:
    """An HTTP server that passes every request to ``api``, one at a time.

    Raises:
        OSError: The address cannot be bound (e.g. the port is in use).
    """

    class Handler(BaseHTTPRequestHandler):
        server_version = "jetank-pose-review"

        def do_GET(self) -> None:
            self._dispatch()

        def do_PUT(self) -> None:
            self._dispatch()

        def do_DELETE(self) -> None:
            self._dispatch()

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("pose review: %s " + format, self.address_string(), *args)

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


def cmd_review_poses(settings: JetankSettings, directory: Path | None, host: str, port: int) -> int:
    """``jetank poses review``: serve the review page until Ctrl-C.

    Raises:
        JetankError: The records directory does not exist, or the port cannot be opened.
    """
    store = GraspDirectory(directory or settings.teach.directory)
    if not store.directory.is_dir():
        raise JetankError(f"no grasp records directory at {store.directory}")
    try:
        server = make_server(PoseReviewApi(store, load_page()), host, port)
    except OSError as exc:
        raise JetankError(f"could not listen on {host}:{port}: {exc}") from exc
    with server, raise_on_termination_signals():
        print(
            f"pose review at http://{socket.gethostname()}.local:{server.server_port}/ "
            f"for the records in {store.directory} (Ctrl-C stops)"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt as exc:
            logger.info("pose review stopped: %s", exc or "Ctrl-C")
    return 0
