"""
Fake neural-network runtime and camera model, for tests and simulation.

Layer: drivers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jetank.domain.perception import FrameAnalysis
from jetank.errors import DriverClosedError, InferenceError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from jetank.domain.models import Frame
    from jetank.domain.perception import NetworkOutputs


@dataclass
class FakeInferenceEngine:
    """:class:`~jetank.interfaces.InferenceEngine` that returns fixed outputs.

    Attributes:
        outputs: Returned by every :meth:`infer`.
        input_shape: The input shape it claims to need.
        batches: Every input it was given.
        fail_with: If set, :meth:`infer` raises this.
        closed: Whether :meth:`close` was called.
    """

    outputs: NetworkOutputs
    input_shape: tuple[int, ...] = (1, 3, 640, 640)
    batches: list[NDArray[np.float32]] = field(default_factory=list)
    fail_with: InferenceError | None = None
    closed: bool = False

    def infer(self, batch: NDArray[np.float32]) -> NetworkOutputs:
        if self.closed:
            raise DriverClosedError("inference engine is closed")
        if batch.shape != self.input_shape:
            raise ValueError(f"input must have shape {self.input_shape}, got {batch.shape}")
        if self.fail_with is not None:
            raise self.fail_with
        self.batches.append(batch)
        return self.outputs

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeFrameAnalyzer:
    """:class:`~jetank.interfaces.FrameAnalyzer` that replays prepared results.

    Each :meth:`analyze` returns the next entry of ``analyses`` (with the frame's timestamp),
    or an empty analysis once they are used up.

    Attributes:
        analyses: Results to hand out, in order.
        fail_with: If set, :meth:`analyze` raises this.
        closed: Whether :meth:`close` was called.
    """

    analyses: list[FrameAnalysis] = field(default_factory=list)
    fail_with: InferenceError | None = None
    closed: bool = False

    def analyze(self, frame: Frame) -> FrameAnalysis:
        if self.closed:
            raise DriverClosedError("frame analyzer is closed")
        if self.fail_with is not None:
            raise self.fail_with
        if not self.analyses:
            return FrameAnalysis(timestamp_s=frame.timestamp_s)
        return self.analyses.pop(0).model_copy(update={"timestamp_s": frame.timestamp_s})

    def close(self) -> None:
        self.closed = True
