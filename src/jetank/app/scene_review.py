"""
Scene review: a web page for fixing the labels of recorded scenes and deleting bad ones.

Layer: app

See :mod:`jetank.app.review_server` for the server and its JSON API. The listing holds every
scene oldest first, so a run of automatically recorded scenes stays together and can be
selected and relabelled in one go, plus the summary ``jetank scenes`` prints and the label
choices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from jetank.app import review_server
from jetank.app.review_server import RecordReviewApi
from jetank.app.scene_commands import format_summary
from jetank.domain.scenes import Lighting, SceneLabels, SceneRecord, summarize_scenes
from jetank.drivers.scene_directory import SceneDirectory
from jetank.errors import JetankError

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.interfaces.scene_store import SceneStore

DEFAULT_PORT: Final = 8082


class SceneListing(BaseModel):
    """What the page shows.

    Attributes:
        records: All scenes, oldest first.
        summary: The same text ``jetank scenes`` prints (empty when there are no scenes).
        lighting: The kinds of light a scene can be labelled with.
        rooms: Rooms used so far, alphabetically, as suggestions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[SceneRecord]
    summary: list[str]
    lighting: list[Lighting]
    rooms: list[str]


def load_page() -> bytes:
    """The scene review page shipped with the package."""
    return review_server.load_page("scene_review.html")


class SceneReviewApi(RecordReviewApi[SceneLabels]):
    """The scene review's requests, answered from a :class:`SceneStore`.

    Args:
        store: Where the scenes are.
        page: HTML returned for ``GET /``.
    """

    def __init__(self, store: SceneStore, page: bytes) -> None:
        super().__init__(store, SceneLabels, page, "scene")
        self._store = store

    def listing(self) -> SceneListing:
        records = self._store.load_all()
        summary = format_summary(summarize_scenes(records)) if records else []
        return SceneListing(
            records=records,
            summary=summary,
            lighting=list(Lighting),
            rooms=sorted({record.labels.room for record in records}),
        )


def cmd_review_scenes(
    settings: JetankSettings, directory: Path | None, host: str, port: int
) -> int:
    """``jetank scenes review``: serve the scene review page until Ctrl-C.

    Raises:
        JetankError: The scenes directory does not exist, or the port cannot be opened.
    """
    store = SceneDirectory(directory if directory is not None else settings.scenes.directory)
    if not store.directory.is_dir():
        raise JetankError(f"no scenes directory at {store.directory}")
    api = SceneReviewApi(store, load_page())
    return review_server.serve(api, host, port, "scene review", store.directory)
