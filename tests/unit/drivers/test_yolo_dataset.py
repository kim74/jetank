from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from jetank.domain.dataset import Split
from jetank.drivers.yolo_dataset import YoloDatasetDirectory
from jetank.errors import StorageError

if TYPE_CHECKING:
    from pathlib import Path

ID = "20261121-172308-331070"


def test_reset_writes_the_layout_and_data_yaml(tmp_path: Path) -> None:
    dataset = YoloDatasetDirectory(tmp_path / "dataset")
    dataset.reset(["toy_animal", "floor"])
    root = tmp_path / "dataset"
    for part in ("images", "labels"):
        for split in ("train", "val"):
            assert (root / part / split).is_dir()
    data = yaml.safe_load((root / "data.yaml").read_text())
    assert data == {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "toy_animal", 1: "floor"},
    }


def test_add_writes_image_and_labels(tmp_path: Path) -> None:
    dataset = YoloDatasetDirectory(tmp_path)
    dataset.reset(["floor"])
    dataset.add(ID, Split.VAL, b"jpeg", ["0 0.1 0.2 0.3 0.4 0.5 0.6"])
    dataset.add("20261121-172310-000000", Split.TRAIN, b"jpeg", [])
    assert (tmp_path / "images" / "val" / f"{ID}.jpg").read_bytes() == b"jpeg"
    assert (tmp_path / "labels" / "val" / f"{ID}.txt").read_text() == "0 0.1 0.2 0.3 0.4 0.5 0.6\n"
    assert (tmp_path / "labels" / "train" / "20261121-172310-000000.txt").read_text() == ""
    with pytest.raises(StorageError, match="not a scene ID"):
        dataset.add("../../x", Split.TRAIN, b"jpeg", [])


def test_a_rebuild_starts_from_scratch(tmp_path: Path) -> None:
    dataset = YoloDatasetDirectory(tmp_path)
    dataset.reset(["floor"])
    dataset.add(ID, Split.TRAIN, b"old", [])
    dataset.reset(["floor"])
    assert list((tmp_path / "images" / "train").iterdir()) == []


def test_refuses_to_clear_a_folder_it_did_not_build(tmp_path: Path) -> None:
    (tmp_path / "precious.txt").write_text("keep me")
    with pytest.raises(StorageError, match="not built by jetank"):
        YoloDatasetDirectory(tmp_path).reset(["floor"])
    assert (tmp_path / "precious.txt").read_text() == "keep me"
    assert not (tmp_path / "images").exists()


def test_unwritable_folder_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(StorageError, match="could not prepare"):
        YoloDatasetDirectory(blocker / "dataset").reset(["floor"])
    with pytest.raises(StorageError, match="could not add"):
        YoloDatasetDirectory(blocker).add(ID, Split.TRAIN, b"jpeg", [])
