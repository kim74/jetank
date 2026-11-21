"""
The folder the labelling tool works in: scene images, ``classes.txt`` and what was drawn.

Layer: drivers

X-AnyLabeling (like LabelMe, whose file format it uses) opens a folder of images and saves
what is drawn on ``<name>.jpg`` as ``<name>.json`` next to it. The images are named by scene
ID, so each drawing can be matched with its scene. ``classes.txt`` lists the class names, one
per line, for the tool's label list.

The drawn JSON files are the operator's work: this module never changes one. Adding an image
never replaces one that is already there, and a draft (a drawing proposed by the model) is only
written where there is no drawing yet.

A draft is not training data until the operator has looked at it. ``.prelabels.json``
remembers a fingerprint (SHA-256) of every draft as written: a draft whose file is unchanged
and not ticked as *checked* in the tool is still unreviewed. Saving a correction, or ticking
the image's check box, makes it count.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jetank.domain.dataset import DrawnImage, DrawnShape
from jetank.domain.records import is_record_id
from jetank.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

CLASSES_FILE: Final = "classes.txt"
DRAFTS_FILE: Final = ".prelabels.json"
#: Format version written into drafts: the X-AnyLabeling version the format was checked with.
LABELME_VERSION: Final = "4.3.3"


# The tool's files carry many more fields (flags, imageData, group IDs, ...); only those
# needed here are read, the rest is ignored.
class _LabelMeShape(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    points: list[tuple[float, float]]
    shape_type: str = "polygon"


class _LabelMeFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shapes: list[_LabelMeShape] = Field(default_factory=list)
    imageWidth: int  # noqa: N815 (the file format's name)
    imageHeight: int  # noqa: N815


class LabellingDirectory:
    """Scene images for the labelling tool, and the drawings it saved.

    Args:
        directory: The folder; created when the first image is added.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def write_classes(self, names: Sequence[str]) -> None:
        """Write ``classes.txt`` with ``names`` in order (the class index order).

        Raises:
            StorageError: The file cannot be written.
        """
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            (self._directory / CLASSES_FILE).write_text("\n".join(names) + "\n", encoding="utf-8")
        except OSError as exc:
            raise StorageError(
                f"could not write {CLASSES_FILE} in {self._directory}: {exc}"
            ) from exc

    def add_image(self, record_id: str, jpeg: bytes) -> bool:
        """Add a scene's image, unless an image of that scene is there already.

        Returns:
            ``True`` if the image was added, ``False`` if it was there already.

        Raises:
            StorageError: ``record_id`` is not a record ID, or the image cannot be written.
        """
        if not is_record_id(record_id):
            raise StorageError(f"not a scene ID: {record_id!r}")
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with (self._directory / f"{record_id}.jpg").open("xb") as image:
                image.write(jpeg)
        except FileExistsError:
            return False
        except OSError as exc:
            raise StorageError(
                f"could not add {record_id}.jpg to {self._directory}: {exc}"
            ) from exc
        return True

    def image_ids(self) -> list[str]:
        """The scene IDs of the images in the folder, in order.

        Raises:
            StorageError: The folder cannot be listed.
        """
        try:
            return sorted(p.stem for p in self._directory.glob("*.jpg") if is_record_id(p.stem))
        except OSError as exc:
            raise StorageError(f"could not list {self._directory}: {exc}") from exc

    def has_drawing(self, record_id: str) -> bool:
        """Whether something was drawn (or drafted) for this scene."""
        return (self._directory / f"{record_id}.json").exists()

    def save_draft(self, drawing: DrawnImage) -> bool:
        """Write a drawing proposed by the model, unless the scene has a drawing already.

        Returns:
            ``True`` if the draft was written, ``False`` if there was a drawing.

        Raises:
            StorageError: A file cannot be written.
        """
        shapes: list[dict[str, object]] = [
            {
                "label": shape.label,
                "points": [list(point) for point in shape.points],
                "group_id": None,
                "description": "",
                "shape_type": shape.kind,
                "flags": {},
            }
            for shape in drawing.shapes
        ]
        content = json.dumps(
            {
                "version": LABELME_VERSION,
                "flags": {},
                "checked": False,
                "shapes": shapes,
                "imagePath": f"{drawing.record_id}.jpg",
                "imageData": None,
                "imageHeight": drawing.height_px,
                "imageWidth": drawing.width_px,
            },
            indent=2,
        ).encode()
        try:
            with (self._directory / f"{drawing.record_id}.json").open("xb") as file:
                file.write(content)
        except FileExistsError:
            return False
        except OSError as exc:
            raise StorageError(f"could not write the draft of {drawing.record_id}: {exc}") from exc
        drafts = self._drafts()
        drafts[drawing.record_id] = hashlib.sha256(content).hexdigest()
        path = self._directory / DRAFTS_FILE
        try:
            temporary = path.with_name(f"{path.name}.tmp")
            temporary.write_text(json.dumps(drafts, indent=1, sort_keys=True), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            raise StorageError(f"could not update {DRAFTS_FILE}: {exc}") from exc
        return True

    def unreviewed_drafts(self) -> set[str]:
        """Scenes whose drawing is still the model's draft, untouched and not checked.

        Raises:
            StorageError: A drawing or the draft list cannot be read.
        """
        unreviewed = set()
        for record_id, fingerprint in self._drafts().items():
            path = self._directory / f"{record_id}.json"
            try:
                content = path.read_bytes()
            except FileNotFoundError:
                continue  # deleted in the tool: nothing to train on anyway
            except OSError as exc:
                raise StorageError(f"could not read {path.name}: {exc}") from exc
            if hashlib.sha256(content).hexdigest() == fingerprint:
                unreviewed.add(record_id)
        return unreviewed

    def _drafts(self) -> dict[str, str]:
        path = self._directory / DRAFTS_FILE
        try:
            drafts = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise StorageError(f"could not read {path}: {exc}") from exc
        if not isinstance(drafts, dict):
            raise StorageError(f"{path} is not a list of drafts")
        return {str(k): str(v) for k, v in drafts.items()}

    def load_drawings(self) -> list[DrawnImage]:
        """Every saved drawing, by scene ID. Images without a drawing are not labelled yet.

        Raises:
            StorageError: The folder cannot be read, or a drawing is unreadable or not named
                after a scene (the message names the file, so it can be fixed).
        """
        try:
            paths = sorted(self._directory.glob("*.json"))
        except OSError as exc:
            raise StorageError(f"could not list {self._directory}: {exc}") from exc
        drawings = []
        for path in paths:
            if path.name.startswith("."):
                continue  # our own bookkeeping, e.g. the draft list
            if not is_record_id(path.stem):
                raise StorageError(f"{path.name} is not named after a scene ID")
            try:
                raw = _LabelMeFile.model_validate_json(path.read_bytes())
                drawings.append(
                    DrawnImage(
                        record_id=path.stem,
                        width_px=raw.imageWidth,
                        height_px=raw.imageHeight,
                        shapes=tuple(
                            DrawnShape(
                                label=shape.label.strip(),
                                kind=shape.shape_type,
                                points=tuple(shape.points),
                            )
                            for shape in raw.shapes
                        ),
                    )
                )
            except (OSError, ValidationError) as exc:
                raise StorageError(f"could not read the drawing {path.name}: {exc}") from exc
        logger.debug("read %d drawings from %s", len(drawings), self._directory)
        return drawings
