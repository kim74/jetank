"""
Port for a camera that delivers BGR frames.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.models import Frame


@runtime_checkable
class Camera(Protocol):
    """Blocking frame source. ``close()`` must be idempotent."""

    def read(self) -> Frame:
        """Capture and return the next frame.

        Raises:
            CameraError: No frame could be captured.
        """
        ...

    def close(self) -> None:
        """Release the camera."""
        ...
