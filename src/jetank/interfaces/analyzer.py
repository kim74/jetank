"""
Port for the camera model: what is in a frame, objects and floor.

Layer: interfaces

Filled today by a YOLO segmentation model on the GPU. The behaviour only sees
:class:`~jetank.domain.perception.FrameAnalysis`, so another model (or a LiDAR for the floor
part) can take its place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.models import Frame
    from jetank.domain.perception import FrameAnalysis


@runtime_checkable
class FrameAnalyzer(Protocol):
    def analyze(self, frame: Frame) -> FrameAnalysis:
        """The objects and floor in ``frame``.

        Raises:
            InferenceError: The model could not be run.
        """
        ...

    def close(self) -> None:
        """Release model resources. Must be idempotent."""
        ...
