from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jetank.control.drive import DriveController
from jetank.control.gripper import GripperController
from jetank.control.teleop import TeleopSession
from jetank.domain.models import TrackSpeeds
from jetank.domain.servo import GripperCalibration
from jetank.domain.status import RobotMode
from jetank.domain.teleop import (
    DriveMessage,
    GripperAction,
    GripperMessage,
    GripperReport,
    Role,
    StopMessage,
)
from jetank.drivers.fake import FakeClock, FakeMotorDriver, FakeServo, FakeServoBus
from jetank.errors import HardwareError, ProtocolError, ServoError

if TYPE_CHECKING:
    from jetank.config.settings import DriveConfig

FORWARD = DriveMessage(type="drive", linear=1.0, angular=0.0)
STOP = StopMessage(type="stop")


@pytest.fixture
def session(motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig) -> TeleopSession:
    return TeleopSession(DriveController(motors, clock, drive_config))


def test_first_client_drives_and_later_ones_observe(session: TeleopSession) -> None:
    assert session.connect("a").role is Role.DRIVER
    assert session.connect("b").role is Role.OBSERVER
    assert session.driver == "a"


def test_mode_reports_whether_someone_is_driving(session: TeleopSession) -> None:
    assert session.mode is RobotMode.TELEOP_WAITING
    session.connect("a")
    assert session.mode is RobotMode.TELEOP_DRIVING
    session.disconnect("a")
    assert session.mode is RobotMode.TELEOP_WAITING


def test_driver_command_reaches_motors_with_speed_limit(
    session: TeleopSession, motors: FakeMotorDriver
) -> None:
    session.connect("a")
    status = session.handle("a", FORWARD)
    assert motors.current == TrackSpeeds(left=0.5, right=0.5)  # drive_config.max_speed = 0.5
    assert status.moving
    assert status.speeds == TrackSpeeds(left=0.5, right=0.5)
    assert status.error is None


def test_observer_cannot_drive(session: TeleopSession, motors: FakeMotorDriver) -> None:
    session.connect("a")
    session.connect("b")
    before = list(motors.history)
    status = session.handle("b", FORWARD)
    assert motors.history == before
    assert status.error == "another client is driving"
    assert not status.moving


def test_any_client_can_stop(session: TeleopSession, motors: FakeMotorDriver) -> None:
    session.connect("a")
    session.connect("b")
    session.handle("a", FORWARD)
    status = session.handle("b", STOP)
    assert motors.current.is_stopped
    assert not status.moving
    assert status.speeds == TrackSpeeds.stopped()


def test_driver_disconnect_stops_and_frees_the_seat(
    session: TeleopSession, motors: FakeMotorDriver
) -> None:
    session.connect("a")
    session.connect("b")
    session.handle("a", FORWARD)
    session.disconnect("a")
    assert motors.current.is_stopped
    assert session.driver is None
    assert session.connect("c").role is Role.DRIVER
    assert session.status("b").role is Role.OBSERVER


def test_observer_disconnect_does_not_stop_driver(
    session: TeleopSession, motors: FakeMotorDriver
) -> None:
    session.connect("a")
    session.connect("b")
    session.handle("a", FORWARD)
    session.disconnect("b")
    assert not motors.current.is_stopped
    assert session.driver == "a"


def test_invalid_message_from_driver_stops(session: TeleopSession, motors: FakeMotorDriver) -> None:
    session.connect("a")
    session.handle("a", FORWARD)
    status = session.reject("a", "invalid message")
    assert motors.current.is_stopped
    assert status.error == "invalid message"


def test_invalid_message_from_observer_does_not_stop_driver(
    session: TeleopSession, motors: FakeMotorDriver
) -> None:
    session.connect("a")
    session.connect("b")
    session.handle("a", FORWARD)
    session.reject("b", "invalid message")
    assert not motors.current.is_stopped


def test_watchdog_stops_a_silent_driver(
    session: TeleopSession, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    session.connect("a")
    session.handle("a", FORWARD)
    clock.advance(0.4)
    assert not session.tick()
    clock.advance(0.2)  # 0.6 s since the last command > 0.5 s timeout
    assert session.tick()
    assert motors.current.is_stopped
    status = session.status("a")
    assert not status.moving
    assert status.speeds == TrackSpeeds.stopped()


def test_repeated_commands_keep_the_robot_moving(
    session: TeleopSession, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    session.connect("a")
    for _ in range(10):
        session.handle("a", FORWARD)
        clock.advance(0.1)
        assert not session.tick()
    assert not motors.current.is_stopped


def test_unknown_client_is_rejected(session: TeleopSession) -> None:
    with pytest.raises(ProtocolError, match="not connected"):
        session.handle("ghost", STOP)
    with pytest.raises(ProtocolError, match="not connected"):
        session.reject("ghost", "invalid message")


def test_duplicate_client_id_is_rejected(session: TeleopSession) -> None:
    session.connect("a")
    with pytest.raises(ProtocolError, match="already connected"):
        session.connect("a")


def test_motor_failure_propagates_after_emergency_stop(
    clock: FakeClock, drive_config: DriveConfig
) -> None:
    class FailingMotors(FakeMotorDriver):
        def set_speeds(self, speeds: TrackSpeeds) -> None:
            raise HardwareError("bus gone")

    motors = FailingMotors()
    session = TeleopSession(DriveController(motors, clock, drive_config))
    session.connect("a")
    with pytest.raises(HardwareError):
        session.handle("a", FORWARD)
    assert motors.current.is_stopped


# --- gripper -------------------------------------------------------------------------------


def gripper_message(action: str, opening: float | None = None) -> GripperMessage:
    return GripperMessage(type="gripper", action=GripperAction(action), opening=opening)


@pytest.fixture
def servo_bus() -> FakeServoBus:
    return FakeServoBus(servos={4: FakeServo(position=855)})


@pytest.fixture
def gripper_session(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig, servo_bus: FakeServoBus
) -> TeleopSession:
    calibration = GripperCalibration(servo_id=4, open_position=745, closed_position=965)
    gripper = GripperController(servo_bus, calibration, move_time_s=0.8, max_torque=0.5)
    return TeleopSession(DriveController(motors, clock, drive_config), gripper)


def test_driver_operates_the_gripper(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    status = gripper_session.handle("a", gripper_message("open"))
    assert status.error is None
    assert status.gripper == GripperReport(engaged=True, opening=1.0)
    assert servo_bus.servos[4].position == 745
    gripper_session.handle("a", gripper_message("set", 0.5))
    assert servo_bus.servos[4].position == 855
    status = gripper_session.handle("a", gripper_message("release"))
    assert status.gripper == GripperReport(engaged=False)
    assert not servo_bus.servos[4].torque_enabled


def test_gripper_torque_is_limited_before_it_is_switched_on(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    gripper_session.handle("a", gripper_message("close"))
    assert servo_bus.torque_limits == [(4, 0.5)]
    assert servo_bus.servos[4].torque_limit == 0.5


def test_observer_cannot_operate_the_gripper(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    gripper_session.connect("b")
    status = gripper_session.handle("b", gripper_message("open"))
    assert status.error == "another client is driving"
    assert servo_bus.moves == []


def test_driver_disconnect_releases_the_gripper(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    gripper_session.handle("a", gripper_message("close"))
    gripper_session.disconnect("a")
    assert not servo_bus.servos[4].torque_enabled
    assert gripper_session.status("b").gripper == GripperReport(engaged=False)


def test_stop_does_not_drop_what_the_gripper_holds(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    gripper_session.handle("a", gripper_message("close"))
    gripper_session.handle("a", STOP)
    assert servo_bus.servos[4].torque_enabled


def test_gripper_failure_is_reported_and_driving_continues(
    gripper_session: TeleopSession, servo_bus: FakeServoBus, motors: FakeMotorDriver
) -> None:
    gripper_session.connect("a")
    servo_bus.fail_with = ServoError("servo 4 does not answer")
    status = gripper_session.handle("a", gripper_message("open"))
    assert status.error == "gripper failed: servo 4 does not answer"
    assert status.gripper == GripperReport(engaged=False)
    assert gripper_session.handle("a", FORWARD).moving
    assert not motors.current.is_stopped
    gripper_session.disconnect("a")  # release fails too, but must not raise
    assert motors.current.is_stopped


def test_without_gripper_status_has_none_and_commands_are_refused(
    session: TeleopSession,
) -> None:
    session.connect("a")
    status = session.handle("a", gripper_message("open"))
    assert status.gripper is None
    assert status.error == "this robot has no gripper"
