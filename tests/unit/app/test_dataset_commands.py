from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
import yaml

from jetank.app import dataset_commands
from jetank.app.cli import main
from jetank.domain.dataset import DrawnImage, DrawnShape
from jetank.domain.records import record_id_for
from jetank.domain.scenes import Lighting, SceneLabels, SceneRecord
from jetank.drivers.labelling_directory import LabellingDirectory
from jetank.drivers.scene_directory import SceneDirectory

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 11, 21, 17, 0, tzinfo=timezone.utc)
FLOOR = {"label": "floor", "points": [[0, 480], [0, 200], [640, 200]], "shape_type": "polygon"}
TOY = {"label": "toy_animal", "points": [[40, 250], [100, 300]], "shape_type": "rectangle"}


def record(seconds: int, animals: bool) -> SceneRecord:
    moment = T0 + timedelta(seconds=seconds)
    labels = SceneLabels(room="kitchen", lighting=Lighting.LAMP, has_animals=animals)
    return SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=labels)


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    """30 scenes (one run, every 2 s; the first 10 with animals) and the CLI arguments."""
    store = SceneDirectory(tmp_path / "scenes")
    for i in range(30):
        store.save(record(2 * i, animals=i < 10), f"jpeg-{i}".encode())
    return {
        "scenes": tmp_path / "scenes",
        "labelling": tmp_path / "labelling",
        "dataset": tmp_path / "dataset",
    }


def draw(folders: dict[str, Path], index: int, *shapes: dict[str, object]) -> None:
    stem = record_id_for(T0 + timedelta(seconds=2 * index))
    drawing = {"shapes": list(shapes), "imageWidth": 640, "imageHeight": 480, "imageData": None}
    (folders["labelling"] / f"{stem}.json").write_text(json.dumps(drawing))


def export(folders: dict[str, Path]) -> int:
    return main(
        [
            "scenes",
            "export",
            "--directory",
            str(folders["scenes"]),
            "--to",
            str(folders["labelling"]),
        ]
    )


def build(folders: dict[str, Path]) -> int:
    return main(
        [
            "dataset",
            "build",
            "--scenes",
            str(folders["scenes"]),
            "--labelling",
            str(folders["labelling"]),
            "--to",
            str(folders["dataset"]),
        ]
    )


def test_export_copies_images_and_keeps_drawings(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert export(folders) == 0
    assert "30 new images" in capsys.readouterr().out
    labelling = folders["labelling"]
    assert len(list(labelling.glob("*.jpg"))) == 30
    assert (labelling / "classes.txt").read_text() == "toy_animal\nfloor\n"
    assert not list(labelling.glob("*.json"))  # the scene records stay where they are

    draw(folders, 0, FLOOR)
    drawing = next(labelling.glob("*.json")).read_text()
    assert export(folders) == 0
    assert "(30 were there already" in capsys.readouterr().out
    assert next(labelling.glob("*.json")).read_text() == drawing


def test_export_needs_scenes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scenes", "export", "--directory", str(tmp_path / "none")]) == 1
    assert "no scenes directory" in capsys.readouterr().err


def test_build_writes_labelled_scenes_split_by_block(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    for i in range(30):
        draw(folders, i, FLOOR, *([TOY] if i < 10 else []))
    assert build(folders) == 0
    out = capsys.readouterr().out
    assert "30 images: 20 train, 10 val" in out
    assert "toy_animal 10, floor 30" in out
    assert "0 warnings" in out

    root = folders["dataset"]
    val = sorted(p.stem for p in (root / "images" / "val").iterdir())
    assert val == [record_id_for(T0 + timedelta(seconds=2 * i)) for i in range(20, 30)]
    first = record_id_for(T0)
    assert (root / "images" / "train" / f"{first}.jpg").read_bytes() == b"jpeg-0"
    assert (root / "labels" / "train" / f"{first}.txt").read_text().splitlines()[0].startswith("1 ")
    assert yaml.safe_load((root / "data.yaml").read_text())["names"] == {
        0: "toy_animal",
        1: "floor",
    }


def test_build_leaves_out_unlabelled_scenes_and_reports_warnings(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    draw(folders, 0, FLOOR)  # recorded with animals, none outlined
    draw(folders, 15, TOY)  # no floor, and recorded without animals
    assert build(folders) == 0
    out = capsys.readouterr().out
    assert "2 images" in out
    assert "28 scenes not labelled yet" in out
    assert "recorded with toy animals, but none is outlined" in out
    assert "no floor outlined" in out
    assert "3 warnings" in out


def test_build_with_errors_writes_nothing(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    draw(folders, 0, FLOOR, {**TOY, "label": "cow"})
    assert build(folders) == 1
    err = capsys.readouterr().err
    assert "unknown class" in err
    assert "1 errors in the drawings: nothing written" in err
    assert not folders["dataset"].exists()


def test_a_drawing_of_a_deleted_scene_is_an_error(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    draw(folders, 3, FLOOR, TOY)
    SceneDirectory(folders["scenes"]).delete(record_id_for(T0 + timedelta(seconds=6)))
    assert build(folders) == 1
    assert "no recorded scene" in capsys.readouterr().err


def test_a_scene_without_image_is_left_out(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    draw(folders, 12, FLOOR)
    (folders["scenes"] / f"{record_id_for(T0 + timedelta(seconds=24))}.jpg").unlink()
    assert build(folders) == 1
    captured = capsys.readouterr()
    assert "no image any more" in captured.out
    assert "no labelled scenes" in captured.err


def test_build_with_nothing_labelled_says_so(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    assert build(folders) == 1
    assert "30 scenes are waiting to be labelled" in capsys.readouterr().err
    assert not folders["dataset"].exists()


def test_build_needs_the_labelling_folder(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert build(folders) == 1
    assert "run `jetank scenes export` first" in capsys.readouterr().err


def test_long_issue_lists_are_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dataset_commands, "MAX_ISSUES_SHOWN", 2)
    from jetank.domain.dataset import LabelIssue, Severity

    issues = [
        LabelIssue(record_id=f"id{i}", severity=Severity.WARNING, message="m") for i in range(3)
    ]
    issues.append(LabelIssue(record_id="id9", severity=Severity.ERROR, message="bad"))
    lines = dataset_commands.format_issues(issues)
    assert lines == ["id9  error: bad", "id0  warning: m", "... and 2 more"]


def val_stems(folders: dict[str, Path]) -> set[str]:
    return {p.stem for p in (folders["dataset"] / "images" / "val").iterdir()}


def test_sparse_labels_reach_val_and_keep_their_side_when_labelling_more(
    folders: dict[str, Path],
) -> None:
    export(folders)
    for i in range(0, 30, 3):  # every 3rd frame of the 60 s run, as when skipping near-twins
        draw(folders, i, FLOOR, *([TOY] if i < 10 else []))
    assert build(folders) == 0
    sparse = val_stems(folders)
    assert sparse, "a run labelled sparsely must still contribute to validation"

    for i in range(30):
        if i % 3:
            draw(folders, i, FLOOR, *([TOY] if i < 10 else []))
    assert build(folders) == 0
    assert sparse <= val_stems(folders)


def test_build_leaves_out_drafts_nobody_reviewed(
    folders: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    export(folders)
    draw(folders, 0, FLOOR, TOY)
    labelling = LabellingDirectory(folders["labelling"])
    floor = DrawnShape(label="floor", kind="polygon", points=((0, 480), (0, 200), (640, 200)))
    for i in (1, 2):
        stem = record_id_for(T0 + timedelta(seconds=2 * i))
        labelling.save_draft(
            DrawnImage(record_id=stem, width_px=640, height_px=480, shapes=(floor,))
        )
    assert build(folders) == 0
    out = capsys.readouterr().out
    assert "1 images" in out
    assert "27 scenes not labelled yet, 2 drafts not reviewed yet" in out
