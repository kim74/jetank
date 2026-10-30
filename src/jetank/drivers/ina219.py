"""
Read-only driver for the TI INA219 power monitor, used for the battery voltage.

Layer: drivers

Only the bus-voltage register is read; nothing is written, so the chip keeps its power-on
configuration (continuous conversion, 32 V range). On the JETANK the shunt is not in the
motor current path, so current and power readings would be meaningless and are not offered.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from jetank.domain.status import BatteryState
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.settings import PowerConfig
    from jetank.interfaces.i2c import I2CBus

logger = logging.getLogger(__name__)

REG_BUS_VOLTAGE: Final = 0x02
#: Bits 15..3 of the bus-voltage register hold the voltage in 4 mV steps (datasheet 8.6.3.2).
BUS_VOLTAGE_SHIFT: Final = 3
BUS_VOLTAGE_LSB_V: Final = 0.004


def bus_voltage_v(raw: int) -> float:
    """Bus voltage in volts from the raw 16-bit bus-voltage register."""
    if not 0 <= raw <= 0xFFFF:
        raise ValueError(f"register value must be 16 bits, got {raw}")
    return (raw >> BUS_VOLTAGE_SHIFT) * BUS_VOLTAGE_LSB_V


class Ina219PowerMonitor:
    """:class:`~jetank.interfaces.PowerMonitor` reading the battery voltage from an INA219.

    Use :meth:`open` to create one on a real bus.

    Args:
        bus: I2C bus the chip is attached to.
        address: 7-bit I2C address.
        owned_bus: Pass ``bus`` again if this driver should close it.
    """

    def __init__(self, bus: I2CBus, address: int, owned_bus: I2CBus | None = None) -> None:
        self._bus = bus
        self._address = address
        self._owned_bus = owned_bus
        self._closed = False

    @classmethod
    def open(cls, config: PowerConfig) -> Ina219PowerMonitor:
        """Open the I2C bus from ``config``. The chip itself is first touched by a reading.

        Raises:
            HardwareError: The bus cannot be opened.
        """
        from smbus2 import SMBus

        try:
            bus = SMBus(config.i2c_bus)
        except OSError as exc:
            raise HardwareError(f"cannot open /dev/i2c-{config.i2c_bus}: {exc}") from exc
        return cls(bus, config.i2c_address, owned_bus=bus)

    def __enter__(self) -> Ina219PowerMonitor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def read_battery(self) -> BatteryState:
        if self._closed:
            raise DriverClosedError("power monitor is closed")
        try:
            high, low = self._bus.read_i2c_block_data(self._address, REG_BUS_VOLTAGE, 2)
        except OSError as exc:
            raise HardwareError(f"INA219 at 0x{self._address:02X} not responding: {exc}") from exc
        return BatteryState(voltage_v=bus_voltage_v((high << 8) | low))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned_bus is not None:
            self._owned_bus.close()
