"""
In-memory grasp store for tests and simulation.

Layer: drivers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jetank.errors import StorageError

if TYPE_CHECKING:
    from jetank.domain.teaching import GraspRecord


@dataclass
class FakeGraspStore:
    """:class:`~jetank.interfaces.GraspStore` that keeps records and images in memory.

    Attributes:
        records: Stored records, oldest first.
        images: Stored JPEG bytes by record ID.
        fail_with: If set, every call raises this.
    """

    records: list[GraspRecord] = field(default_factory=list)
    images: dict[str, bytes] = field(default_factory=dict)
    fail_with: StorageError | None = None

    def save(self, record: GraspRecord, jpeg: bytes | None) -> GraspRecord:
        self._check()
        if any(r.record_id == record.record_id for r in self.records):
            raise StorageError(f"record {record.record_id} already exists")
        if jpeg is not None:
            self.images[record.record_id] = jpeg
            record = record.model_copy(update={"image_file": f"{record.record_id}.jpg"})
        self.records.append(record)
        return record

    def load_all(self) -> list[GraspRecord]:
        self._check()
        return list(self.records)

    def _check(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with
