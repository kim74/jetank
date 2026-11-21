from __future__ import annotations

import numpy as np
import pytest

from jetank.domain.models import Frame
from jetank.domain.perception import FrameAnalysis, NetworkOutputs
from jetank.drivers.fake import FakeFrameAnalyzer, FakeInferenceEngine
from jetank.errors import DriverClosedError, InferenceError
from jetank.interfaces import FrameAnalyzer, InferenceEngine

OUTPUTS = NetworkOutputs(tensors={"y": np.zeros((1, 6, 4), np.float32)})
FRAME = Frame(image=np.zeros((4, 4, 3), np.uint8), timestamp_s=3.0)


def test_fake_engine_checks_its_input_and_returns_its_outputs() -> None:
    engine = FakeInferenceEngine(outputs=OUTPUTS, input_shape=(1, 3, 8, 8))
    assert isinstance(engine, InferenceEngine)
    assert engine.infer(np.zeros((1, 3, 8, 8), np.float32)) is OUTPUTS
    with pytest.raises(ValueError, match="shape"):
        engine.infer(np.zeros((1, 3, 4, 4), np.float32))
    engine.fail_with = InferenceError("boom")
    with pytest.raises(InferenceError):
        engine.infer(np.zeros((1, 3, 8, 8), np.float32))
    engine.close()
    with pytest.raises(DriverClosedError):
        engine.infer(np.zeros((1, 3, 8, 8), np.float32))


def test_fake_analyzer_replays_results_with_the_frame_time() -> None:
    prepared = FrameAnalysis(timestamp_s=0.0, detections=())
    analyzer = FakeFrameAnalyzer(analyses=[prepared])
    assert isinstance(analyzer, FrameAnalyzer)
    assert analyzer.analyze(FRAME).timestamp_s == 3.0
    assert analyzer.analyze(FRAME) == FrameAnalysis(timestamp_s=3.0)
    analyzer.fail_with = InferenceError("boom")
    with pytest.raises(InferenceError):
        analyzer.analyze(FRAME)
    analyzer.close()
    with pytest.raises(DriverClosedError):
        analyzer.analyze(FRAME)
