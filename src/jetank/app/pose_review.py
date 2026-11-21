"""
Pose review: a web page for looking through taught grasp records, fixing labels, deleting bad ones.

Layer: app

See :mod:`jetank.app.review_server` for the server and its JSON API. The listing holds every
record (newest first), the fit summary and suggested poses, and the label choices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from jetank.app import review_server
from jetank.app.arm_commands import format_fit, format_poses
from jetank.app.review_server import RecordReviewApi
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
from jetank.errors import JetankError

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.interfaces.grasp_store import GraspStore

DEFAULT_PORT: Final = 8081


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
    return review_server.load_page("pose_review.html")


class PoseReviewApi(RecordReviewApi[TeachLabels]):
    """The pose review's requests, answered from a :class:`GraspStore`.

    Args:
        store: Where the records are.
        page: HTML returned for ``GET /``.
    """

    def __init__(self, store: GraspStore, page: bytes) -> None:
        super().__init__(store, TeachLabels, page, "grasp record")
        self._store = store

    def listing(self) -> ReviewListing:
        records = self._store.load_all()
        summary = [*format_fit(summarize_fit(records)), "", *format_poses(records)]
        return ReviewListing(records=records[::-1], summary=summary, choices=_CHOICES)


def cmd_review_poses(settings: JetankSettings, directory: Path | None, host: str, port: int) -> int:
    """``jetank poses review``: serve the review page until Ctrl-C.

    Raises:
        JetankError: The records directory does not exist, or the port cannot be opened.
    """
    store = GraspDirectory(directory or settings.teach.directory)
    if not store.directory.is_dir():
        raise JetankError(f"no grasp records directory at {store.directory}")
    api = PoseReviewApi(store, load_page())
    return review_server.serve(api, host, port, "pose review", store.directory)
