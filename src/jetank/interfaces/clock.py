"""
Port for a monotonic time source, so timing logic can be tested deterministically.

Layer: interfaces
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def monotonic(self) -> float:
        """Seconds from an arbitrary but fixed origin; never goes backwards."""
        ...
