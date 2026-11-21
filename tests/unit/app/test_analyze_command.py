from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from jetank.app.analyze_command import cmd_analyze, describe
from jetank.app.cli import main
from jetank.app.wiring import build_frame_analyzer
from jetank.config.settings import DetectorConfig, load_settings
from jetank.domain.detection import BoundingBox, Detection
from jetank.domain.perception import FloorMask, FrameAnalysis
from jetank.drivers.fake import FakeFrameAnalyzer
from jetank.errors import InferenceError, JetankError

if TYPE_CHECKING:
    from pathlib import Path

cv2 = pytest.importorskip("cv2")


def fake_settings(tmp_path: Path) -> Path:
    config = tmp_path / "jetank.yaml"
    config.write_text("detector:\n  backend: fake\n", encoding="utf-8")
    return config


def test_analyze_prints_findings_and_writes_annotated_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "scene.jpg"
    cv2.imwrite(str(image), np.full((48, 64, 3), 90, dtype=np.uint8))
    out = tmp_path / "out"
    args = ["--config", str(fake_settings(tmp_path)), "analyze", str(image), str(image)]
    assert main([*args, "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "scene.jpg: nothing" in printed
    assert "2 images" in printed
    assert (out / "scene.jpg").is_file()


def test_unreadable_images_are_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--config", str(fake_settings(tmp_path)), "analyze", str(tmp_path / "missing.jpg")]
    assert main(args) == 1
    assert "could not read the image" in capsys.readouterr().err


def test_analyze_needs_images(tmp_path: Path) -> None:
    with pytest.raises(JetankError, match="no images"):
        cmd_analyze(load_settings(fake_settings(tmp_path)), [], None)


def test_describe_lists_detections_and_floor() -> None:
    analysis = FrameAnalysis(
        timestamp_s=0.0,
        detections=(
            Detection(label="toy_animal", confidence=0.87, box=BoundingBox(x1=0, y1=0, x2=1, y2=1)),
        ),
        floor=FloorMask(cells=np.array([[False, True]])),
    )
    assert describe(analysis) == "toy_animal 0.87; floor 50% of the frame"


def test_wiring_builds_the_configured_analyzer(tmp_path: Path) -> None:
    assert isinstance(build_frame_analyzer(DetectorConfig(backend="fake")), FakeFrameAnalyzer)
    missing = DetectorConfig(backend="tensorrt", engine_path=tmp_path / "none.engine")
    with pytest.raises(InferenceError, match="no TensorRT engine"):
        build_frame_analyzer(missing)
