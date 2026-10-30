"""
Battery monitor returning a settable voltage.

Layer: drivers (fake)
"""

from __future__ import annotations

from jetank.domain.status import BatteryState
from jetank.errors import DriverClosedError, HardwareError


class FakePowerMonitor:
    """:class:`~jetank.interfaces.PowerMonitor` reporting :attr:`voltage_v`.

    Attributes:
        voltage_v: Voltage returned by the next reading.
        fail_with: If set, readings raise this (to simulate a missing chip).
    """

    def __init__(self, voltage_v: float = 12.0) -> None:
        self.voltage_v = voltage_v
        self.fail_with: HardwareError | None = None
        self.closed = False

    def read_battery(self) -> BatteryState:
        if self.closed:
            raise DriverClosedError("power monitor is closed")
        if self.fail_with is not None:
            raise self.fail_with
        return BatteryState(voltage_v=self.voltage_v)

    def close(self) -> None:
        self.closed = True
