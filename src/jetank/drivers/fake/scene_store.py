"""
In-memory scene store for tests and simulation.

Layer: drivers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jetank.errors import RecordNotFoundError, StorageError

if TYPE_CHECKING:
    from jetank.domain.scenes import SceneLabels, SceneRecord


@dataclass
class FakeSceneStore:
    """:class:`~jetank.interfaces.SceneStore` that keeps scenes and images in memory.

    Attributes:
        records: Stored scenes, oldest first.
        images: Stored JPEG bytes by record ID.
        deleted: Scenes removed with :meth:`delete`, in the order they were removed.
        fail_with: If set, every call raises this.
    """

    records: list[SceneRecord] = field(default_factory=list)
    images: dict[str, bytes] = field(default_factory=dict)
    deleted: list[SceneRecord] = field(default_factory=list)
    fail_with: StorageError | None = None

    def save(self, record: SceneRecord, jpeg: bytes) -> SceneRecord:
        self._check()
        if any(r.record_id == record.record_id for r in self.records):
            raise StorageError(f"scene {record.record_id} already exists")
        self.images[record.record_id] = jpeg
        record = record.model_copy(update={"image_file": f"{record.record_id}.jpg"})
        self.records.append(record)
        return record

    def load_all(self) -> list[SceneRecord]:
        self._check()
        return list(self.records)

    def load_image(self, record_id: str) -> bytes | None:
        self._check()
        self._index(record_id)
        return self.images.get(record_id)

    def relabel(self, record_id: str, labels: SceneLabels) -> SceneRecord:
        self._check()
        index = self._index(record_id)
        self.records[index] = self.records[index].relabeled(labels)
        return self.records[index]

    def delete(self, record_id: str) -> None:
        self._check()
        self.deleted.append(self.records.pop(self._index(record_id)))
        self.images.pop(record_id, None)

    def _index(self, record_id: str) -> int:
        for index, record in enumerate(self.records):
            if record.record_id == record_id:
                return index
        raise RecordNotFoundError(f"no scene {record_id!r}")

    def _check(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with
