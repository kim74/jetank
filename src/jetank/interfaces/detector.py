"""
Port for an object detector.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.detection import Detection
    from jetank.domain.models import Frame


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]:
        """Return every object found in ``frame`` above the configured confidence."""
        ...

    def close(self) -> None:
        """Release model resources. Must be idempotent."""
        ...
