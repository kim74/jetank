"""
Grasp records as files: ``<record_id>.json`` plus ``<record_id>.jpg`` in one directory.

Layer: drivers

File names come only from the record ID, which the domain restricts to digits and dashes,
so nothing an operator types can choose where a file is written. Files are created
exclusively, so a record is never overwritten.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from jetank.domain.teaching import GraspRecord
from jetank.errors import StorageError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            if jpeg is not None:
                image_name = f"{record.record_id}.jpg"
                with (self._directory / image_name).open("xb") as image:
                    image.write(jpeg)
                record = record.model_copy(update={"image_file": image_name})
            with (self._directory / f"{record.record_id}.json").open("x", encoding="utf-8") as f:
                f.write(record.model_dump_json(indent=2))
        except FileExistsError as exc:
            raise StorageError(f"grasp record {record.record_id} already exists") from exc
        except OSError as exc:
            raise StorageError(f"could not save grasp record in {self._directory}: {exc}") from exc
        logger.info("teach: saved %s in %s", record.record_id, self._directory)
        return record

    def load_all(self) -> list[GraspRecord]:
        if not self._directory.exists():
            return []
        try:
            paths = sorted(self._directory.glob("*.json"))
        except OSError as exc:
            raise StorageError(f"could not list {self._directory}: {exc}") from exc
        records = []
        for path in paths:
            try:
                records.append(GraspRecord.model_validate_json(path.read_bytes()))
            except (OSError, ValidationError) as exc:
                logger.warning("teach: skipping unreadable record %s: %s", path.name, exc)
        return sorted(records, key=lambda r: r.recorded_at)
