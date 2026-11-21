"""Real TensorRT engines on the Jetson's GPU. Run with `pytest -m hardware`. Moves nothing.

Needs the engines built by scripts/export_detector.sh; a missing one is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jetank.domain.models import Frame
from jetank.drivers.tensorrt_engine import TensorRtEngine
from jetank.perception.analyzer import YoloAnalyzer
from jetank.perception.yolo import YoloSettings

pytestmark = pytest.mark.hardware

MODELS = Path(__file__).resolve().parents[2] / "models"
FRAME = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=1.0)


def engine_or_skip(name: str) -> Path:
    path = MODELS / name
    if not path.is_file():
        pytest.skip(f"no engine {path}")
    return path


def test_coco_detection_engine_runs() -> None:
    with TensorRtEngine(engine_or_skip("yolo11n_640_fp16.engine")) as engine:
        assert engine.input_shape == (1, 3, 640, 640)
        outputs = engine.infer(np.zeros(engine.input_shape, dtype=np.float32))
        assert outputs.with_rank(3).shape == (1, 84, 8400)


def test_toy_segmentation_engine_finds_nothing_in_a_black_frame() -> None:
    engine = TensorRtEngine(engine_or_skip("toys-seg_640_fp16.engine"))
    settings = YoloSettings(class_names=("toy_animal", "floor"), floor_label="floor")
    with YoloAnalyzer(engine, settings) as analyzer:
        result = analyzer.analyze(FRAME)
    assert result.floor is not None
    assert result.floor.cells.shape == (120, 160)
