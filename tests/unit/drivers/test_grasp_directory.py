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
from jetank.errors import RecordNotFoundError, StorageError

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


def relabelled(animal: str = "lion", result: GraspResult = GraspResult.SHAKY) -> TeachLabels:
    return TeachLabels(
        animal=animal,
        stage=TeachStage.PRE_GRASP,
        posture=ToyPosture.ON_SIDE,
        grip_point=GripPoint.LEGS,
        result=result,
        note="jaws slipped",
    )


def test_the_image_of_a_record_loads_back(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    with_image = store.save(make_record(1), b"\xff\xd8jpeg")
    without = store.save(make_record(2), None)
    assert store.load_image(with_image.record_id) == b"\xff\xd8jpeg"
    assert store.load_image(without.record_id) is None


def test_a_missing_image_file_reads_as_no_image(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), b"jpeg")
    (tmp_path / f"{record.record_id}.jpg").unlink()
    assert store.load_image(record.record_id) is None
    assert "missing" in caplog.text


def test_relabel_changes_only_the_labels_and_persists(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), b"jpeg")
    updated = store.relabel(record.record_id, relabelled())
    assert updated == record.model_copy(update={"labels": relabelled()})
    assert GraspDirectory(tmp_path).load_all() == [updated]
    assert store.load_image(record.record_id) == b"jpeg"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        f"{record.record_id}.jpg",
        f"{record.record_id}.json",
    ]


def test_delete_moves_record_and_image_to_the_trash(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    gone = store.save(make_record(1), b"jpeg")
    kept = store.save(make_record(2), None)
    store.delete(gone.record_id)
    assert store.load_all() == [kept]
    trash = tmp_path / "trash"
    assert sorted(p.name for p in trash.iterdir()) == [
        f"{gone.record_id}.jpg",
        f"{gone.record_id}.json",
    ]
    # Moving both files back restores the record.
    for path in trash.iterdir():
        path.replace(tmp_path / path.name)
    assert store.load_all() == [gone, kept]


def test_delete_without_an_image(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), None)
    store.delete(record.record_id)
    assert store.load_all() == []
    assert [p.name for p in (tmp_path / "trash").iterdir()] == [f"{record.record_id}.json"]


@pytest.mark.parametrize(
    "record_id", ["20261122-090000-000001", "../secrets", "20261122-090000-000000/../x", ""]
)
def test_unknown_or_unsafe_ids_are_not_found(tmp_path: Path, record_id: str) -> None:
    store = GraspDirectory(tmp_path)
    store.save(make_record(), None)
    (tmp_path / "secrets.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RecordNotFoundError):
        store.load_image(record_id)
    with pytest.raises(RecordNotFoundError):
        store.relabel(record_id, relabelled())
    with pytest.raises(RecordNotFoundError):
        store.delete(record_id)
    assert (tmp_path / "secrets.json").exists()


def test_a_corrupt_record_cannot_be_changed(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record_id = make_record().record_id
    (tmp_path / f"{record_id}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError, match="unreadable"):
        store.relabel(record_id, relabelled())


def test_unreadable_files_raise_storage_error(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), None)
    json_path = tmp_path / f"{record.record_id}.json"
    json_path.unlink()
    json_path.mkdir()  # a directory where the file should be: reading it fails
    with pytest.raises(StorageError, match="could not read"):
        store.load_image(record.record_id)


def test_an_unreadable_image_raises_storage_error(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), b"jpeg")
    image = tmp_path / f"{record.record_id}.jpg"
    image.unlink()
    image.mkdir()
    with pytest.raises(StorageError, match="could not read the image"):
        store.load_image(record.record_id)


def test_failed_writes_raise_storage_error(tmp_path: Path) -> None:
    store = GraspDirectory(tmp_path)
    record = store.save(make_record(), None)
    (tmp_path / f"{record.record_id}.json.tmp").mkdir()  # blocks the temporary file
    with pytest.raises(StorageError, match="could not update"):
        store.relabel(record.record_id, relabelled())
    (tmp_path / "trash").write_text("in the way", encoding="utf-8")
    with pytest.raises(StorageError, match="could not delete"):
        store.delete(record.record_id)
    assert store.load_all() == [record]
