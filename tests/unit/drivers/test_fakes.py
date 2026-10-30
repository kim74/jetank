from __future__ import annotations

import pytest

from jetank.domain.models import TrackSpeeds
from jetank.domain.status import StatusScreen
from jetank.drivers.fake import (
    FakeCamera,
    FakeClock,
    FakeDisplay,
    FakeHostInfo,
    FakeI2CBus,
    FakeMotorDriver,
    FakePowerMonitor,
    FakeServo,
    FakeServoBus,
)
from jetank.errors import DriverClosedError, HardwareError, ServoError
from jetank.interfaces import (
    Camera,
    Clock,
    Display,
    HostInfo,
    I2CBus,
    MotorDriver,
    PowerMonitor,
    SerialPort,
    ServoBus,
)


def test_fakes_satisfy_their_interfaces(clock: FakeClock) -> None:
    assert isinstance(clock, Clock)
    assert isinstance(FakeMotorDriver(), MotorDriver)
    assert isinstance(FakeCamera(4, 3, clock), Camera)
    assert isinstance(FakeI2CBus(), I2CBus)
    assert isinstance(FakeDisplay(), Display)
    assert isinstance(FakePowerMonitor(), PowerMonitor)
    assert isinstance(FakeHostInfo(), HostInfo)
    assert isinstance(FakeServoBus(), ServoBus)


class TestFakeClock:
    def test_advance_moves_time(self) -> None:
        clock = FakeClock(start_s=1.0)
        clock.advance(0.25)
        assert clock.monotonic() == 1.25

    def test_cannot_go_backwards(self) -> None:
        with pytest.raises(ValueError, match="backwards"):
            FakeClock().advance(-1.0)


class TestFakeMotorDriver:
    def test_records_commands_and_close_stops(self, motors: FakeMotorDriver) -> None:
        motors.set_speeds(TrackSpeeds(left=0.2, right=0.3))
        motors.close()
        motors.close()
        assert motors.history == [TrackSpeeds(left=0.2, right=0.3), TrackSpeeds.stopped()]

    def test_use_after_close_raises(self, motors: FakeMotorDriver) -> None:
        motors.close()
        with pytest.raises(DriverClosedError):
            motors.set_speeds(TrackSpeeds.stopped())
        with pytest.raises(DriverClosedError):
            motors.stop()


class TestFakeCamera:
    def test_frames_have_configured_size_and_differ(self, clock: FakeClock) -> None:
        camera = FakeCamera(width=8, height=6, clock=clock)
        first = camera.read()
        clock.advance(0.1)
        second = camera.read()
        assert first.image.shape == (6, 8, 3)
        assert second.timestamp_s > first.timestamp_s
        assert not (first.image == second.image).all()

    def test_read_after_close_raises(self, clock: FakeClock) -> None:
        camera = FakeCamera(width=8, height=6, clock=clock)
        camera.close()
        with pytest.raises(DriverClosedError):
            camera.read()

    def test_rejects_empty_size(self, clock: FakeClock) -> None:
        with pytest.raises(ValueError, match="positive"):
            FakeCamera(width=0, height=6, clock=clock)


class TestFakeI2CBus:
    def test_block_write_fills_consecutive_registers(self, i2c_bus: FakeI2CBus) -> None:
        i2c_bus.write_i2c_block_data(0x40, 0x10, [1, 2, 3])
        assert i2c_bus.registers[0x40] == {0x10: 1, 0x11: 2, 0x12: 3}
        assert i2c_bus.read_byte_data(0x40, 0x11) == 2

    def test_rejects_non_byte_values(self, i2c_bus: FakeI2CBus) -> None:
        with pytest.raises(ValueError, match="out of range"):
            i2c_bus.write_byte_data(0x40, 0, 256)

    def test_simulated_failure_raises(self, i2c_bus: FakeI2CBus) -> None:
        i2c_bus.fail_with = OSError(121, "Remote I/O error")
        with pytest.raises(OSError, match="Remote I/O"):
            i2c_bus.read_byte_data(0x40, 0)

    def test_use_after_close_raises(self, i2c_bus: FakeI2CBus) -> None:
        i2c_bus.close()
        with pytest.raises(DriverClosedError):
            i2c_bus.write_byte_data(0x40, 0, 0)

    def test_block_read_returns_consecutive_registers(self, i2c_bus: FakeI2CBus) -> None:
        i2c_bus.write_i2c_block_data(0x41, 0x02, [0x5D, 0x3A])
        assert i2c_bus.read_i2c_block_data(0x41, 0x02, 2) == [0x5D, 0x3A]

    def test_block_writes_are_logged_and_limited_to_32_bytes(self, i2c_bus: FakeI2CBus) -> None:
        i2c_bus.write_i2c_block_data(0x3C, 0x40, [1, 2])
        assert i2c_bus.block_writes == [(0x3C, 0x40, [1, 2])]
        with pytest.raises(ValueError, match="at most 32"):
            i2c_bus.write_i2c_block_data(0x3C, 0x40, [0] * 33)


class TestFakeDisplay:
    def test_records_screens_and_close_blanks(self) -> None:
        display = FakeDisplay()
        display.show(StatusScreen(lines=("hi",)))
        display.clear()
        display.close()
        display.close()
        assert [s.lines for s in display.history] == [("hi",), (), ()]

    def test_use_after_close_raises(self) -> None:
        display = FakeDisplay()
        display.close()
        with pytest.raises(DriverClosedError):
            display.show(StatusScreen(lines=()))

    def test_simulated_failure_raises(self) -> None:
        display = FakeDisplay()
        display.fail_with = HardwareError("gone")
        with pytest.raises(HardwareError):
            display.clear()


class TestFakePowerMonitor:
    def test_reports_set_voltage(self) -> None:
        power = FakePowerMonitor(voltage_v=11.1)
        assert power.read_battery().voltage_v == 11.1

    def test_failure_and_close(self) -> None:
        power = FakePowerMonitor()
        power.fail_with = HardwareError("gone")
        with pytest.raises(HardwareError):
            power.read_battery()
        power.close()
        with pytest.raises(DriverClosedError):
            power.read_battery()


class TestFakeServoBus:
    def test_goal_is_reached_only_with_torque(self) -> None:
        bus = FakeServoBus(servos={4: FakeServo(position=500, goal=500)})
        bus.move(4, 600, 0.5)
        assert bus.read_state(4).position == 500
        bus.set_torque(4, True)
        assert bus.read_state(4).position == 600

    def test_close_switches_torque_off(self) -> None:
        bus = FakeServoBus(servos={4: FakeServo()})
        bus.set_torque(4, True)
        bus.close()
        bus.close()
        assert bus.torque_log == [(4, True), (4, False)]
        with pytest.raises(DriverClosedError):
            bus.ping(4)

    def test_missing_servo_and_bad_values(self) -> None:
        bus = FakeServoBus(servos={4: FakeServo()})
        with pytest.raises(ServoError):
            bus.read_state(3)
        with pytest.raises(ValueError, match="position"):
            bus.move(4, 2000, 0.5)
        with pytest.raises(ValueError, match="duration"):
            bus.move(4, 500, 0.0)
        bus.fail_with = ServoError("bus down")
        with pytest.raises(ServoError):
            bus.ping(4)


def test_scs_test_port_satisfies_serial_interface() -> None:
    from tests.unit.drivers.scs_helpers import FakeScsPort

    assert isinstance(FakeScsPort(), SerialPort)
