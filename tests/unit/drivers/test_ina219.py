from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jetank.drivers.ina219 import REG_BUS_VOLTAGE, Ina219PowerMonitor, bus_voltage_v
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from jetank.drivers.fake import FakeI2CBus

ADDRESS = 0x41


def set_bus_register(bus: FakeI2CBus, raw: int) -> None:
    bus.registers[ADDRESS][REG_BUS_VOLTAGE] = raw >> 8
    bus.registers[ADDRESS][REG_BUS_VOLTAGE + 1] = raw & 0xFF


@pytest.mark.parametrize(
    ("raw", "volts"),
    [
        (0x0000, 0.0),
        (0x5D3A, 2983 * 0.004),  # 11.932 V; CNVR (bit 1) set, ignored
        (0x5D38, 2983 * 0.004),
        (0xFFF8, 8191 * 0.004),  # full scale 32.764 V
    ],
)
def test_bus_voltage_uses_bits_15_to_3_at_4mv(raw: int, volts: float) -> None:
    assert bus_voltage_v(raw) == pytest.approx(volts)


def test_bus_voltage_rejects_wider_values() -> None:
    with pytest.raises(ValueError, match="16 bits"):
        bus_voltage_v(0x10000)


def test_reads_big_endian_register(i2c_bus: FakeI2CBus) -> None:
    set_bus_register(i2c_bus, 0x5D3A)
    battery = Ina219PowerMonitor(i2c_bus, ADDRESS).read_battery()
    assert battery.voltage_v == pytest.approx(11.932)


def test_never_writes_to_the_chip(i2c_bus: FakeI2CBus) -> None:
    set_bus_register(i2c_bus, 0x5D3A)
    with Ina219PowerMonitor(i2c_bus, ADDRESS) as monitor:
        monitor.read_battery()
    assert i2c_bus.writes == []


def test_missing_chip_raises_hardware_error(i2c_bus: FakeI2CBus) -> None:
    i2c_bus.fail_with = OSError(121, "Remote I/O error")
    with pytest.raises(HardwareError, match="0x41 not responding"):
        Ina219PowerMonitor(i2c_bus, ADDRESS).read_battery()


def test_close_releases_owned_bus_once(i2c_bus: FakeI2CBus) -> None:
    monitor = Ina219PowerMonitor(i2c_bus, ADDRESS, owned_bus=i2c_bus)
    monitor.close()
    monitor.close()
    assert i2c_bus.closed
    with pytest.raises(DriverClosedError):
        monitor.read_battery()


def test_close_leaves_shared_bus_open(i2c_bus: FakeI2CBus) -> None:
    Ina219PowerMonitor(i2c_bus, ADDRESS).close()
    assert not i2c_bus.closed
