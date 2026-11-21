from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from jetank.domain.records import record_id_for
from jetank.domain.scenes import Lighting, SceneLabels, SceneRecord
from jetank.drivers.fake import FakeSceneStore
from jetank.drivers.scene_directory import SceneDirectory
from jetank.errors import RecordNotFoundError, StorageError
from jetank.interfaces import SceneStore

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)
LABELS = SceneLabels(room="kitchen", lighting=Lighting.LAMP)


def make_scene(seconds: int = 0) -> SceneRecord:
    moment = T0 + timedelta(seconds=seconds)
    return SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=LABELS)


def test_saves_json_and_image_and_loads_them_oldest_first(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path / "scenes")
    later = store.save(make_scene(5), b"jpeg-later")
    earlier = store.save(make_scene(0), b"jpeg-earlier")
    assert later.image_file == f"{later.record_id}.jpg"
    assert (tmp_path / "scenes" / later.image_file).read_bytes() == b"jpeg-later"
    assert store.load_all() == [earlier, later]


def test_never_overwrites_a_scene(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path)
    store.save(make_scene(), b"first")
    with pytest.raises(StorageError, match="already exists"):
        store.save(make_scene(), b"second")
    assert (tmp_path / f"{make_scene().record_id}.jpg").read_bytes() == b"first"


def test_a_missing_directory_holds_no_scenes(tmp_path: Path) -> None:
    assert SceneDirectory(tmp_path / "nothing").load_all() == []


def test_unreadable_files_are_skipped(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path)
    saved = store.save(make_scene(), b"jpeg")
    (tmp_path / "20261122-090001-000000.json").write_text("{not json", encoding="utf-8")
    assert store.load_all() == [saved]


def test_an_unwritable_directory_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(StorageError, match="could not save"):
        SceneDirectory(blocker / "scenes").save(make_scene(), b"jpeg")


def test_both_stores_implement_the_interface(tmp_path: Path) -> None:
    assert isinstance(SceneDirectory(tmp_path), SceneStore)
    assert isinstance(FakeSceneStore(), SceneStore)


def test_fake_store_keeps_scenes_and_can_fail() -> None:
    store = FakeSceneStore()
    saved = store.save(make_scene(), b"jpeg")
    assert store.images[saved.record_id] == b"jpeg"
    assert store.load_all() == [saved]
    with pytest.raises(StorageError):
        store.save(make_scene(), b"again")
    store.fail_with = StorageError("disk full")
    with pytest.raises(StorageError, match="disk full"):
        store.load_all()


def test_relabel_rewrites_only_the_labels(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path)
    saved = store.save(make_scene(), b"jpeg")
    new = SceneLabels(room="hall", lighting=Lighting.DIM, has_animals=False)
    updated = store.relabel(saved.record_id, new)
    assert updated == saved.model_copy(update={"labels": new})
    assert store.load_all() == [updated]
    assert store.load_image(saved.record_id) == b"jpeg"
    assert not list(tmp_path.glob("*.tmp"))


def test_delete_moves_the_scene_to_the_trash(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path)
    saved = store.save(make_scene(), b"jpeg")
    store.delete(saved.record_id)
    assert store.load_all() == []
    assert sorted(p.name for p in (tmp_path / "trash").iterdir()) == [
        f"{saved.record_id}.jpg",
        f"{saved.record_id}.json",
    ]


@pytest.mark.parametrize("record_id", ["20261122-090009-000000", "../etc/passwd", "x"])
def test_unknown_or_unsafe_ids_are_not_found(tmp_path: Path, record_id: str) -> None:
    store = SceneDirectory(tmp_path)
    store.save(make_scene(), b"jpeg")
    with pytest.raises(RecordNotFoundError, match="no scene"):
        store.load_image(record_id)
    with pytest.raises(RecordNotFoundError):
        store.relabel(record_id, LABELS)
    with pytest.raises(RecordNotFoundError):
        store.delete(record_id)


def test_a_missing_image_reads_as_none(tmp_path: Path) -> None:
    store = SceneDirectory(tmp_path)
    saved = store.save(make_scene(), b"jpeg")
    (tmp_path / f"{saved.record_id}.jpg").unlink()
    assert store.load_image(saved.record_id) is None


def test_fake_store_relabels_and_deletes() -> None:
    store = FakeSceneStore()
    saved = store.save(make_scene(), b"jpeg")
    new = SceneLabels(room="hall", lighting=Lighting.DIM)
    assert store.relabel(saved.record_id, new).labels == new
    assert store.load_image(saved.record_id) == b"jpeg"
    store.delete(saved.record_id)
    assert store.records == []
    assert [r.labels for r in store.deleted] == [new]
    with pytest.raises(RecordNotFoundError):
        store.load_image(saved.record_id)
