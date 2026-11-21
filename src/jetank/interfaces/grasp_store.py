"""
Port for storing taught grasp poses: written by the teaching controller, reviewed later.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.teaching import GraspRecord, TeachLabels


@runtime_checkable
class GraspStore(Protocol):
    def save(self, record: GraspRecord, jpeg: bytes | None) -> GraspRecord:
        """Persist ``record`` and the camera image taken with it, if any.

        Returns:
            The record as stored, with ``image_file`` set when an image was saved.

        Raises:
            StorageError: The record could not be written, or its ID is already taken.
        """
        ...

    def load_all(self) -> list[GraspRecord]:
        """Every stored record, oldest first. Unreadable entries are skipped with a warning.

        Raises:
            StorageError: The store itself cannot be read.
        """
        ...

    def load_image(self, record_id: str) -> bytes | None:
        """The JPEG saved with a record, or ``None`` if the record has no image.

        Raises:
            RecordNotFoundError: There is no record with this ID.
            StorageError: The record or its image cannot be read.
        """
        ...

    def relabel(self, record_id: str, labels: TeachLabels) -> GraspRecord:
        """Replace a record's labels; its measurements and image stay as they were.

        Returns:
            The updated record.

        Raises:
            RecordNotFoundError: There is no record with this ID.
            StorageError: The record cannot be read or written.
        """
        ...

    def delete(self, record_id: str) -> None:
        """Remove a record and its image from the store (a store may keep them recoverable).

        Raises:
            RecordNotFoundError: There is no record with this ID.
            StorageError: The record cannot be removed.
        """
        ...
