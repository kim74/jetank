"""
Port for the track motor driver.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.models import TrackSpeeds


@runtime_checkable
class MotorDriver(Protocol):
    """Sets the speed of the two tracks.

    Implementations must make ``close()`` idempotent and leave the motors stopped.
    """

    def set_speeds(self, speeds: TrackSpeeds) -> None:
        """Apply normalised speeds to the left and right track."""
        ...

    def stop(self) -> None:
        """Stop both tracks immediately."""
        ...

    def close(self) -> None:
        """Stop the motors and release the underlying bus."""
        ...
