"""
The camera model: a YOLO network on an :class:`~jetank.interfaces.InferenceEngine`.

Layer: perception
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.errors import InferenceError
from jetank.perception.letterbox import letterbox
from jetank.perception.yolo import decode

if TYPE_CHECKING:
    from jetank.domain.detection import Detection
    from jetank.domain.models import Frame
    from jetank.domain.perception import FrameAnalysis
    from jetank.interfaces.inference import InferenceEngine
    from jetank.perception.yolo import YoloSettings

logger = logging.getLogger(__name__)


class YoloAnalyzer:
    """:class:`~jetank.interfaces.FrameAnalyzer` (and :class:`~jetank.interfaces.Detector`)
    for an Ultralytics YOLO model, detection or segmentation.

    Args:
        engine: Runs the network; its input must be ``(1, 3, height, width)``. Owned: closed
            with the analyzer.
        settings: Class names and thresholds.

    Raises:
        InferenceError: The engine's input is not a single RGB image.
    """

    def __init__(self, engine: InferenceEngine, settings: YoloSettings) -> None:
        shape = engine.input_shape
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            engine.close()
            raise InferenceError(f"expected a (1, 3, height, width) input, the model has {shape}")
        self._engine = engine
        self._settings = settings
        self._input_height, self._input_width = shape[2], shape[3]

    def analyze(self, frame: Frame) -> FrameAnalysis:
        batch, box = letterbox(frame.image, self._input_width, self._input_height)
        outputs = self._engine.infer(batch)
        try:
            return decode(outputs, box, self._settings, frame.timestamp_s)
        except ValueError as exc:
            raise InferenceError(f"the model's output does not fit the config: {exc}") from exc

    def detect(self, frame: Frame) -> list[Detection]:
        return list(self.analyze(frame).detections)

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> YoloAnalyzer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
