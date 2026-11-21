"""
Records as files: ``<record_id>.json`` plus an optional ``<record_id>.jpg`` in one directory.

Layer: drivers

Shared by the grasp and scene stores. File names come only from the record ID, which the
domain restricts to digits and dashes, so nothing an operator types can choose where a file
is read or written. New records are created exclusively, so saving never overwrites one;
relabelling rewrites the JSON file atomically. Deleting moves both files into the ``trash``
subdirectory, so a record deleted by mistake can be moved back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, TypeVar

from pydantic import ValidationError

from jetank.domain.records import ImageRecord, is_record_id
from jetank.errors import RecordNotFoundError, StorageError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

R = TypeVar("R", bound=ImageRecord)

#: Subdirectory that deleted records are moved into.
TRASH_DIRECTORY: Final = "trash"


def save_new(directory: Path, record: R, jpeg: bytes | None, kind: str) -> R:
    """Write a new record and its image, creating ``directory`` if needed.

    Args:
        directory: Where the files go.
        record: The record; its ``image_file`` is set here when ``jpeg`` is given.
        jpeg: The camera image, or ``None``.
        kind: What the record is, for messages (e.g. "grasp record").

    Returns:
        The record as written.

    Raises:
        StorageError: A record with this ID exists already, or a file cannot be written.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if jpeg is not None:
            image_name = f"{record.record_id}.jpg"
            with (directory / image_name).open("xb") as image:
                image.write(jpeg)
            record = record.model_copy(update={"image_file": image_name})
        with (directory / f"{record.record_id}.json").open("x", encoding="utf-8") as file:
            file.write(record.model_dump_json(indent=2))
    except FileExistsError as exc:
        raise StorageError(f"{kind} {record.record_id} already exists") from exc
    except OSError as exc:
        raise StorageError(f"could not save {kind} in {directory}: {exc}") from exc
    logger.info("saved %s %s in %s", kind, record.record_id, directory)
    return record


def load_all(directory: Path, model: type[R], kind: str) -> list[R]:
    """Every record in ``directory``, oldest first; unreadable files are skipped with a warning.

    A directory that does not exist yet holds no records.

    Raises:
        StorageError: The directory cannot be listed.
    """
    if not directory.exists():
        return []
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        raise StorageError(f"could not list {directory}: {exc}") from exc
    records = []
    for path in paths:
        try:
            records.append(model.model_validate_json(path.read_bytes()))
        except (OSError, ValidationError) as exc:
            logger.warning("skipping unreadable %s %s: %s", kind, path.name, exc)
    return sorted(records, key=lambda r: r.recorded_at)


def load_one(directory: Path, record_id: str, model: type[R], kind: str) -> R:
    """The record ``record_id`` in ``directory``.

    Raises:
        RecordNotFoundError: There is no such record, or ``record_id`` is not a record ID.
        StorageError: The record cannot be read or parsed.
    """
    path = json_path(directory, record_id, kind)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise RecordNotFoundError(f"no {kind} {record_id!r}") from exc
    except OSError as exc:
        raise StorageError(f"could not read {kind} {record_id}: {exc}") from exc
    try:
        return model.model_validate_json(data)
    except ValidationError as exc:
        raise StorageError(f"{kind} {record_id} is unreadable: {exc}") from exc


def load_image(directory: Path, record: ImageRecord, kind: str) -> bytes | None:
    """The JPEG saved with ``record``, or ``None`` if it has none (or the file has gone).

    Raises:
        StorageError: The image exists but cannot be read.
    """
    if record.image_file is None:
        return None
    try:
        return image_path(directory, record.record_id, kind).read_bytes()
    except FileNotFoundError:
        logger.warning("image of %s %s is missing", kind, record.record_id)
        return None
    except OSError as exc:
        raise StorageError(f"could not read the image of {record.record_id}: {exc}") from exc


def rewrite(directory: Path, record: ImageRecord, kind: str) -> None:
    """Replace the JSON file of an existing record atomically (old or new, never half a file).

    Raises:
        StorageError: The file cannot be written.
    """
    path = json_path(directory, record.record_id, kind)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise StorageError(f"could not update {kind} {record.record_id}: {exc}") from exc


def move_to_trash(directory: Path, record_id: str, kind: str) -> Path:
    """Move a record's JSON and image into ``directory/trash``, so it can be moved back.

    Returns:
        The trash directory.

    Raises:
        StorageError: The files cannot be moved.
    """
    trash = directory / TRASH_DIRECTORY
    image = image_path(directory, record_id, kind)
    try:
        trash.mkdir(exist_ok=True)
        if image.exists():
            image.replace(trash / image.name)
        path = json_path(directory, record_id, kind)
        path.replace(trash / path.name)
    except OSError as exc:
        raise StorageError(f"could not delete {kind} {record_id}: {exc}") from exc
    return trash


def json_path(directory: Path, record_id: str, kind: str) -> Path:
    """Where the JSON of ``record_id`` lives.

    Raises:
        RecordNotFoundError: ``record_id`` is not a record ID. This check is what keeps a
            request from reaching files outside the directory.
    """
    if not is_record_id(record_id):
        raise RecordNotFoundError(f"no {kind} {record_id!r}")
    return directory / f"{record_id}.json"


def image_path(directory: Path, record_id: str, kind: str) -> Path:
    """Where the image of ``record_id`` lives (see :func:`json_path` for the ID check)."""
    return json_path(directory, record_id, kind).with_suffix(".jpg")
