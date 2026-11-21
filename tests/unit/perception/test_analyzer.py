from __future__ import annotations

import numpy as np
import pytest

from jetank.domain.models import Frame
from jetank.drivers.fake import FakeInferenceEngine
from jetank.errors import InferenceError
from jetank.interfaces import Detector, FrameAnalyzer
from jetank.perception.analyzer import YoloAnalyzer
from jetank.perception.yolo import YoloSettings

from .yolo_outputs import CLASSES, Candidate, outputs

SETTINGS = YoloSettings(class_names=CLASSES, floor_label="floor")
FRAME = Frame(image=np.zeros((480, 640, 3), dtype=np.uint8), timestamp_s=7.0)


def engine(*candidates: Candidate, classes: int = 2) -> FakeInferenceEngine:
    return FakeInferenceEngine(outputs=outputs(*candidates, classes=classes))


def test_analyzes_a_frame_through_the_engine() -> None:
    fake = engine(Candidate(320, 320, 64, 48, (0.9, 0.0)))
    analyzer = YoloAnalyzer(fake, SETTINGS)
    result = analyzer.analyze(FRAME)
    assert result.timestamp_s == 7.0
    assert [d.label for d in result.detections] == ["toy_animal"]
    assert result.floor is not None
    assert fake.batches[0].shape == (1, 3, 640, 640)
    assert [d.label for d in analyzer.detect(FRAME)] == ["toy_animal"]


def test_implements_both_ports() -> None:
    analyzer = YoloAnalyzer(engine(), SETTINGS)
    assert isinstance(analyzer, FrameAnalyzer)
    assert isinstance(analyzer, Detector)


def test_a_model_that_does_not_fit_the_config_raises_inference_error() -> None:
    analyzer = YoloAnalyzer(engine(classes=80), SETTINGS)
    with pytest.raises(InferenceError, match="does not fit the config"):
        analyzer.analyze(FRAME)


def test_an_engine_without_an_image_input_is_refused_and_closed() -> None:
    fake = FakeInferenceEngine(outputs=outputs(), input_shape=(1, 640, 640))
    with pytest.raises(InferenceError, match="input"):
        YoloAnalyzer(fake, SETTINGS)
    assert fake.closed


def test_engine_failures_pass_through_and_closing_closes_the_engine() -> None:
    fake = engine()
    fake.fail_with = InferenceError("GPU gone")
    with YoloAnalyzer(fake, SETTINGS) as analyzer, pytest.raises(InferenceError, match="GPU"):
        analyzer.analyze(FRAME)
    assert fake.closed
