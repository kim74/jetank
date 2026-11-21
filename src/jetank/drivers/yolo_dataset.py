"""
A YOLO segmentation dataset as files, ready for ``yolo segment train data=<dir>/data.yaml``.

Layer: drivers

Layout::

    <dir>/data.yaml                    class names and where the parts are
    <dir>/images/{train,val}/<id>.jpg
    <dir>/labels/{train,val}/<id>.txt  one line per outline (see jetank.domain.dataset)
    <dir>/.jetank-dataset              marks the folder as built here

The dataset is always rebuilt from scratch. Only a folder carrying the marker (or an empty
or missing one) is cleared first, so pointing the build at the wrong folder cannot delete
anything else.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING, Final

import yaml

from jetank.domain.dataset import Split
from jetank.domain.records import is_record_id
from jetank.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

MARKER: Final = ".jetank-dataset"
DATA_FILE: Final = "data.yaml"
_PARTS: Final = ("images", "labels")


class YoloDatasetDirectory:
    """Writes a YOLO segmentation dataset into a folder.

    Args:
        directory: The dataset folder.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def reset(self, class_names: Sequence[str]) -> None:
        """Empty the dataset and write ``data.yaml`` for ``class_names`` (in index order).

        Raises:
            StorageError: The folder holds files but was not built here, or it cannot be
                cleared or written.
        """
        directory = self._directory
        try:
            if directory.exists() and any(directory.iterdir()):
                if not (directory / MARKER).exists():
                    raise StorageError(
                        f"{directory} is not empty and was not built by jetank; "
                        "choose another folder or empty it"
                    )
                for part in _PARTS:
                    if (directory / part).exists():
                        shutil.rmtree(directory / part)
            for part in _PARTS:
                for split in Split:
                    (directory / part / split.value).mkdir(parents=True, exist_ok=True)
            (directory / MARKER).write_text("built by `jetank dataset build`\n", encoding="utf-8")
            data = {
                "path": str(directory.resolve()),
                "train": f"images/{Split.TRAIN.value}",
                "val": f"images/{Split.VAL.value}",
                "names": dict(enumerate(class_names)),
            }
            (directory / DATA_FILE).write_text(
                yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
            )
        except OSError as exc:
            raise StorageError(f"could not prepare the dataset in {directory}: {exc}") from exc

    def add(self, record_id: str, split: Split, jpeg: bytes, lines: Sequence[str]) -> None:
        """Add one image and its label file (no lines: an image with nothing to find).

        Raises:
            StorageError: ``record_id`` is not a record ID, or the files cannot be written.
        """
        if not is_record_id(record_id):
            raise StorageError(f"not a scene ID: {record_id!r}")
        try:
            (self._directory / "images" / split.value / f"{record_id}.jpg").write_bytes(jpeg)
            label = self._directory / "labels" / split.value / f"{record_id}.txt"
            label.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"could not add {record_id} to the dataset: {exc}") from exc
