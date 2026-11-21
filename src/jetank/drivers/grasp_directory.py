"""
Grasp records as files: ``<record_id>.json`` plus ``<record_id>.jpg`` in one directory.

Layer: drivers

File names come only from the record ID, which the domain restricts to digits and dashes,
so nothing an operator types can choose where a file is read or written. New records are
created exclusively, so saving never overwrites one; relabelling rewrites the JSON file
atomically (a crash leaves the old or the new version, never half a file). Deleting moves
both files into the ``trash`` subdirectory, so a record deleted by mistake can be moved back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from jetank.domain.teaching import GraspRecord, is_record_id
from jetank.errors import RecordNotFoundError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.domain.teaching import TeachLabels

logger = logging.getLogger(__name__)

#: Subdirectory that deleted records are moved into.
TRASH_DIRECTORY: Final = "trash"


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

    def load_image(self, record_id: str) -> bytes | None:
        record = self._load(record_id)
        if record.image_file is None:
            return None
        try:
            return self._image_path(record_id).read_bytes()
        except FileNotFoundError:
            logger.warning("teach: image of %s is missing", record_id)
            return None
        except OSError as exc:
            raise StorageError(f"could not read the image of {record_id}: {exc}") from exc

    def relabel(self, record_id: str, labels: TeachLabels) -> GraspRecord:
        record = self._load(record_id).relabeled(labels)
        path = self._json_path(record_id)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            raise StorageError(f"could not update grasp record {record_id}: {exc}") from exc
        logger.info("teach: relabelled %s", record_id)
        return record

    def delete(self, record_id: str) -> None:
        self._load(record_id)
        trash = self._directory / TRASH_DIRECTORY
        image = self._image_path(record_id)
        try:
            trash.mkdir(exist_ok=True)
            if image.exists():
                image.replace(trash / image.name)
            path = self._json_path(record_id)
            path.replace(trash / path.name)
        except OSError as exc:
            raise StorageError(f"could not delete grasp record {record_id}: {exc}") from exc
        logger.info("teach: moved %s to %s", record_id, trash)

    def _load(self, record_id: str) -> GraspRecord:
        path = self._json_path(record_id)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise RecordNotFoundError(f"no grasp record {record_id!r}") from exc
        except OSError as exc:
            raise StorageError(f"could not read grasp record {record_id}: {exc}") from exc
        try:
            return GraspRecord.model_validate_json(data)
        except ValidationError as exc:
            raise StorageError(f"grasp record {record_id} is unreadable: {exc}") from exc

    def _json_path(self, record_id: str) -> Path:
        # The ID check is what keeps a request from reaching files outside the directory.
        if not is_record_id(record_id):
            raise RecordNotFoundError(f"no grasp record {record_id!r}")
        return self._directory / f"{record_id}.json"

    def _image_path(self, record_id: str) -> Path:
        return self._json_path(record_id).with_suffix(".jpg")
