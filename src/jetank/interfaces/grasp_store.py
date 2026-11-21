"""
Port for storing taught grasp poses.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.teaching import GraspRecord


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
