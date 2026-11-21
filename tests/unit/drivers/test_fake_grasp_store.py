from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jetank.domain.teaching import (
    GraspRecord,
    GraspResult,
    GripPoint,
    TeachLabels,
    TeachStage,
    ToyPosture,
)
from jetank.drivers.fake import FakeGraspStore
from jetank.errors import RecordNotFoundError, StorageError


def labels(animal: str = "cow") -> TeachLabels:
    return TeachLabels(
        animal=animal,
        stage=TeachStage.GRASP,
        posture=ToyPosture.STANDING,
        grip_point=GripPoint.BODY,
        result=GraspResult.GOOD,
    )


def make_record(record_id: str = "20261122-090000-000000") -> GraspRecord:
    return GraspRecord(
        record_id=record_id,
        recorded_at=datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc),
        labels=labels(),
        joints={},
    )


def test_fake_store_relabels_and_deletes() -> None:
    store = FakeGraspStore()
    first = store.save(make_record("20261122-090000-000001"), b"jpeg")
    second = store.save(make_record("20261122-090000-000002"), None)
    assert store.load_image(first.record_id) == b"jpeg"
    assert store.load_image(second.record_id) is None
    updated = store.relabel(second.record_id, labels("lion"))
    assert updated.labels.animal == "lion"
    store.delete(first.record_id)
    assert store.load_all() == [updated]
    assert store.deleted == [first]
    assert store.images == {}


def test_fake_store_reports_unknown_ids_and_failures() -> None:
    store = FakeGraspStore()
    with pytest.raises(RecordNotFoundError):
        store.delete("20261122-090000-000009")
    store.fail_with = StorageError("disk full")
    with pytest.raises(StorageError, match="disk full"):
        store.load_image("20261122-090000-000009")
