"""
Grasp records as files: ``<record_id>.json`` plus ``<record_id>.jpg`` in one directory.

Layer: drivers

See :mod:`jetank.drivers.record_files` for how files are named, written, rewritten when
relabelled and moved to ``trash/`` when deleted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from jetank.domain.teaching import GraspRecord
from jetank.drivers import record_files
from jetank.drivers.record_files import TRASH_DIRECTORY

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.domain.teaching import TeachLabels

__all__ = ["TRASH_DIRECTORY", "GraspDirectory"]

logger = logging.getLogger(__name__)

_KIND: Final = "grasp record"


class GraspDirectory:
    """:class:`~jetank.interfaces.GraspStore` backed by a directory.

    Args:
        directory: Where records are written; created on the first save.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def save(self, record: GraspRecord, jpeg: bytes | None) -> GraspRecord:
        return record_files.save_new(self._directory, record, jpeg, _KIND)

    def load_all(self) -> list[GraspRecord]:
        return record_files.load_all(self._directory, GraspRecord, _KIND)

    def load_image(self, record_id: str) -> bytes | None:
        return record_files.load_image(self._directory, self._load(record_id), _KIND)

    def relabel(self, record_id: str, labels: TeachLabels) -> GraspRecord:
        record = self._load(record_id).relabeled(labels)
        record_files.rewrite(self._directory, record, _KIND)
        logger.info("teach: relabelled %s", record_id)
        return record

    def delete(self, record_id: str) -> None:
        self._load(record_id)
        trash = record_files.move_to_trash(self._directory, record_id, _KIND)
        logger.info("teach: moved %s to %s", record_id, trash)

    def _load(self, record_id: str) -> GraspRecord:
        return record_files.load_one(self._directory, record_id, GraspRecord, _KIND)
