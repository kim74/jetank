"""
Port for storing recorded scenes: written while driving around, labelled later for training.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.scenes import SceneLabels, SceneRecord


@runtime_checkable
class SceneStore(Protocol):
    def save(self, record: SceneRecord, jpeg: bytes) -> SceneRecord:
        """Persist ``record`` and its camera image.

        Returns:
            The record as stored, with ``image_file`` set.

        Raises:
            StorageError: The record could not be written, or its ID is already taken.
        """
        ...

    def load_all(self) -> list[SceneRecord]:
        """Every stored scene, oldest first. Unreadable entries are skipped with a warning.

        Raises:
            StorageError: The store itself cannot be read.
        """
        ...

    def load_image(self, record_id: str) -> bytes | None:
        """The JPEG saved with a scene, or ``None`` if it has none.

        Raises:
            RecordNotFoundError: There is no scene with this ID.
            StorageError: The scene or its image cannot be read.
        """
        ...

    def relabel(self, record_id: str, labels: SceneLabels) -> SceneRecord:
        """Replace a scene's labels; its image and arm pose stay as they were.

        Returns:
            The updated scene.

        Raises:
            RecordNotFoundError: There is no scene with this ID.
            StorageError: The scene cannot be read or written.
        """
        ...

    def delete(self, record_id: str) -> None:
        """Remove a scene and its image from the store (a store may keep them recoverable).

        Raises:
            RecordNotFoundError: There is no scene with this ID.
            StorageError: The scene cannot be removed.
        """
        ...
