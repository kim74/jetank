from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.config.settings import DriveConfig
from jetank.control.drive import DriveController
from jetank.domain.models import DriveCommand, TrackSpeeds
from jetank.drivers.fake import FakeClock, FakeMotorDriver
from jetank.errors import HardwareError


@pytest.fixture
def drive(motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig) -> DriveController:
    return DriveController(motors, clock, drive_config)


class FailingMotors(FakeMotorDriver):
    def set_speeds(self, speeds: TrackSpeeds) -> None:
        raise HardwareError("bus gone")


def test_starts_with_motors_stopped(drive: DriveController, motors: FakeMotorDriver) -> None:
    assert motors.history == [TrackSpeeds.stopped()]
    assert not drive.is_moving


def test_command_is_mixed_and_limited(drive: DriveController, motors: FakeMotorDriver) -> None:
    sent = drive.command(DriveCommand(linear=1.0, angular=0.0))
    assert sent == TrackSpeeds(left=0.5, right=0.5)  # max_speed = 0.5
    assert motors.current == sent
    assert drive.is_moving


def test_set_tracks_is_limited(drive: DriveController, motors: FakeMotorDriver) -> None:
    drive.set_tracks(TrackSpeeds(left=-1.0, right=0.5))
    assert motors.current == TrackSpeeds(left=-0.5, right=0.25)


@given(linear=st.floats(-1, 1), angular=st.floats(-1, 1), max_speed=st.floats(0.01, 1.0))
def test_never_exceeds_max_speed(linear: float, angular: float, max_speed: float) -> None:
    motors = FakeMotorDriver()
    drive = DriveController(motors, FakeClock(), DriveConfig(max_speed=max_speed))
    sent = drive.command(DriveCommand(linear=linear, angular=angular))
    assert abs(sent.left) <= max_speed
    assert abs(sent.right) <= max_speed


def test_watchdog_keeps_running_within_timeout(
    drive: DriveController, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    drive.command(DriveCommand(linear=0.5))
    clock.advance(0.5)
    assert drive.tick() is False
    assert not motors.current.is_stopped


def test_watchdog_stops_after_timeout_once(
    drive: DriveController, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    drive.command(DriveCommand(linear=0.5))
    clock.advance(0.51)
    assert drive.tick() is True
    assert motors.current.is_stopped
    assert not drive.is_moving
    stops_so_far = len(motors.history)
    clock.advance(1.0)
    assert drive.tick() is False
    assert len(motors.history) == stops_so_far


def test_new_command_resets_watchdog(
    drive: DriveController, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    drive.command(DriveCommand(linear=0.5))
    clock.advance(0.4)
    drive.command(DriveCommand(linear=0.5))
    clock.advance(0.4)
    assert drive.tick() is False
    assert not motors.current.is_stopped


def test_zero_command_does_not_arm_watchdog(
    drive: DriveController, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    drive.command(DriveCommand())
    clock.advance(5.0)
    assert drive.tick() is False
    assert not drive.is_moving


def test_stop_halts_motors(drive: DriveController, motors: FakeMotorDriver) -> None:
    drive.command(DriveCommand(linear=0.5))
    drive.stop()
    assert motors.current.is_stopped
    assert not drive.is_moving


def test_context_exit_stops_even_on_error(drive: DriveController, motors: FakeMotorDriver) -> None:
    def drive_then_fail() -> None:
        with drive:
            drive.command(DriveCommand(linear=0.5))
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        drive_then_fail()
    assert motors.current.is_stopped


def test_failed_command_triggers_stop_and_propagates(
    clock: FakeClock, drive_config: DriveConfig
) -> None:
    motors = FailingMotors()
    drive = DriveController(motors, clock, drive_config)
    with pytest.raises(HardwareError):
        drive.command(DriveCommand(linear=0.5))
    assert motors.history[-1] == TrackSpeeds.stopped()
    assert not drive.is_moving


def test_failed_emergency_stop_is_logged_not_masking(
    clock: FakeClock, drive_config: DriveConfig, caplog: pytest.LogCaptureFixture
) -> None:
    motors = FailingMotors()
    drive = DriveController(motors, clock, drive_config)
    motors.closed = True  # stop() now raises too
    with pytest.raises(HardwareError, match="bus gone"):
        drive.command(DriveCommand(linear=0.5))
    assert "failed to stop motors" in caplog.text
