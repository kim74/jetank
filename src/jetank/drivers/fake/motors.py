"""
In-memory motor driver that records every command.

Layer: drivers (fake)
"""

from __future__ import annotations

from jetank.domain.models import TrackSpeeds
from jetank.errors import DriverClosedError


class FakeMotorDriver:
    """:class:`~jetank.interfaces.MotorDriver` that stores commands instead of moving anything.

    Attributes:
        history: Every speed pair applied, in order (``stop()`` records a stopped pair).
    """

    def __init__(self) -> None:
        self.history: list[TrackSpeeds] = []
        self.closed = False

    @property
    def current(self) -> TrackSpeeds:
        """The most recently applied speeds (stopped if nothing was applied yet)."""
        return self.history[-1] if self.history else TrackSpeeds.stopped()

    def set_speeds(self, speeds: TrackSpeeds) -> None:
        if self.closed:
            raise DriverClosedError("motor driver is closed")
        self.history.append(speeds)

    def stop(self) -> None:
        if self.closed:
            raise DriverClosedError("motor driver is closed")
        self.history.append(TrackSpeeds.stopped())

    def close(self) -> None:
        if not self.closed:
            self.history.append(TrackSpeeds.stopped())
            self.closed = True
