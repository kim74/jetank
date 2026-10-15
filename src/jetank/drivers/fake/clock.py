"""
Manually advanced clock for deterministic tests and simulation.

Layer: drivers (fake)
"""

from __future__ import annotations


class FakeClock:
    """:class:`~jetank.interfaces.Clock` whose time only moves when :meth:`advance` is called."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._now = start_s

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a monotonic clock cannot go backwards")
        self._now += seconds
