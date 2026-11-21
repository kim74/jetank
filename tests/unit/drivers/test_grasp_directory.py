from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from jetank.domain.servo import JointName, ServoState
from jetank.domain.teaching import (
    GraspRecord,
    GraspResult,
    GripPoint,
    TeachLabels,
    TeachStage,
    ToyPosture,
    record_id_for,
)
from jetank.drivers.grasp_directory import GraspDirectory
from jetank.errors import StorageError

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)


def make_record(seconds: int = 0, animal: str = "cow") -> GraspRecord:
    moment = T0 + timedelta(seconds=seconds)
    base = ServoState(
        servo_id=1,
        position=575,
        voltage_v=5.4,
        temperature_c=30,
        moving=False,
        torque_enabled=True,
    )
    return GraspRecord(
        record_id=record_id_for(moment),
        recorded_at=moment,
        labels=TeachLabels(
            animal=animal,
            stage=TeachStage.GRASP,
            posture=ToyPosture.STANDING,
            grip_point=GripPoint.BODY,
            result=GraspResult.GOOD,
        ),
        joints={JointName.BASE: base},
    )


def test_saved_records_load_back_oldest_first(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path / "grasps")
    later = store.save(make_record(5, "lion"), None)
    earlier = store.save(make_record(1, "cow"), b"\xff\xd8jpeg")
    assert earlier.image_file == f"{earlier.record_id}.jpg"
    assert (tmp_path / "grasps" / earlier.image_file).read_bytes() == b"\xff\xd8jpeg"
    assert later.image_file is None
    assert store.load_all() == [earlier, later]


def test_a_record_is_never_overwritten(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    store.save(make_record(), None)
    with pytest.raises(StorageError, match="already exists"):
        store.save(make_record(), None)


def test_a_missing_directory_has_no_records(tmp_path: Path) -> None:
    assert GraspDirectory(tmp_path / "nothing").load_all() == []


def test_unreadable_records_are_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = GraspDirectory(tmp_path)
    good = store.save(make_record(), None)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert store.load_all() == [good]
    assert "broken.json" in caplog.text


def test_an_unwritable_directory_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("in the way", encoding="utf-8")
    with pytest.raises(StorageError, match="could not save"):
        GraspDirectory(blocker / "grasps").save(make_record(), None)
