from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jetank.app.cli import main
from jetank.app.prelabel_command import prelabel
from jetank.domain.detection import BoundingBox, Detection
from jetank.domain.perception import FloorMask, FrameAnalysis
from jetank.drivers.fake import FakeFrameAnalyzer
from jetank.drivers.labelling_directory import LabellingDirectory
from jetank.errors import JetankError

if TYPE_CHECKING:
    from pathlib import Path

cv2 = pytest.importorskip("cv2")

IDS = ["20261121-172308-000001", "20261121-172308-000002", "20261121-172308-000003"]


def folder(tmp_path: Path) -> LabellingDirectory:
    labelling = LabellingDirectory(tmp_path / "labelling")
    for record_id in IDS:
        _, jpeg = cv2.imencode(".jpg", np.full((480, 640, 3), 120, dtype=np.uint8))
        labelling.add_image(record_id, jpeg.tobytes())
    return labelling


def seen() -> FrameAnalysis:
    cells = np.zeros((120, 160), dtype=bool)
    cells[60:] = True
    toy = Detection(
        label="toy_animal", confidence=0.3, box=BoundingBox(x1=0.4, y1=0.5, x2=0.5, y2=0.6)
    )
    return FrameAnalysis(timestamp_s=0.0, detections=(toy,), floor=FloorMask(cells=cells))


def test_drafts_only_scenes_without_a_drawing(tmp_path: Path) -> None:
    labelling = folder(tmp_path)
    mine = {"shapes": [], "imageWidth": 640, "imageHeight": 480}
    (labelling.directory / f"{IDS[0]}.json").write_text(json.dumps(mine))
    report = prelabel(FakeFrameAnalyzer(analyses=[seen(), seen()]), labelling)
    assert (report.drafted, report.already_drawn, report.toys, report.floor_outlines) == (
        2,
        1,
        2,
        2,
    )
    assert json.loads((labelling.directory / f"{IDS[0]}.json").read_text()) == mine
    assert labelling.unreviewed_drafts() == {IDS[1], IDS[2]}
    drawn = {d.record_id: d for d in labelling.load_drawings()}
    assert [s.label for s in drawn[IDS[1]].shapes] == ["toy_animal", "floor"]
    # a second run has nothing left to draft
    assert prelabel(FakeFrameAnalyzer(), labelling).drafted == 0


def test_cli_prelabel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labelling = folder(tmp_path)
    config = tmp_path / "jetank.yaml"
    config.write_text("detector:\n  backend: fake\n", encoding="utf-8")
    args = ["--config", str(config), "scenes", "prelabel", "--to", str(labelling.directory)]
    assert main(args) == 0
    assert "drafted 3 scenes" in capsys.readouterr().out
    assert main([*args, "--confidence", "1.5"]) == 1
    assert "--confidence" in capsys.readouterr().err
    assert main(["--config", str(config), "scenes", "prelabel", "--to", str(tmp_path / "x")]) == 1
    assert "run `jetank scenes export` first" in capsys.readouterr().err


def test_an_unreadable_image_is_reported(tmp_path: Path) -> None:
    labelling = folder(tmp_path)
    (labelling.directory / f"{IDS[0]}.jpg").write_bytes(b"not a jpeg")
    with pytest.raises(JetankError, match="could not read the image"):
        prelabel(FakeFrameAnalyzer(), labelling)
