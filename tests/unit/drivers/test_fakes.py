from __future__ import annotations

import pytest

from jetank.domain.models import TrackSpeeds
from jetank.drivers.fake import FakeCamera, FakeClock, FakeI2CBus, FakeMotorDriver
from jetank.errors import DriverClosedError
from jetank.interfaces import Camera, Clock, I2CBus, MotorDriver


def test_fakes_satisfy_their_interfaces(clock: FakeClock) -> None:
    assert isinstance(clock, Clock)
    assert isinstance(FakeMotorDriver(), MotorDriver)
    assert isinstance(FakeCamera(4, 3, clock), Camera)
    assert isinstance(FakeI2CBus(), I2CBus)


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
