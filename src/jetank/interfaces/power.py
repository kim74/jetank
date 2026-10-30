"""
Port for the battery monitor.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.status import BatteryState


@runtime_checkable
class PowerMonitor(Protocol):
    """Reads the battery. Implementations must make ``close()`` idempotent."""

    def read_battery(self) -> BatteryState:
        """Take one battery reading.

        Raises:
            HardwareError: The monitor could not be read.
        """
        ...

    def close(self) -> None:
        """Release the underlying bus."""
        ...
