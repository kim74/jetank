from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jetank.config.settings import DriveConfig
from jetank.control.arm import PARK_SETTLE_S, ArmController
from jetank.control.drive import DriveController
from jetank.control.gripper import GripperController
from jetank.control.teleop import TeleopSession
from jetank.domain.models import TrackSpeeds
from jetank.domain.servo import GripperCalibration, JointCalibration, JointName
from jetank.domain.status import RobotMode
from jetank.domain.teleop import (
    ArmJogMessage,
    ArmReport,
    DriveMessage,
    GripperAction,
    GripperMessage,
    GripperReport,
    Role,
    SleepMessage,
    StopMessage,
)
from jetank.drivers.fake import FakeClock, FakeMotorDriver, FakeServo, FakeServoBus
from jetank.errors import HardwareError, ProtocolError, ServoError

if TYPE_CHECKING:
    from jetank.domain.servo import ServoState

FORWARD = DriveMessage(type="drive", linear=1.0, angular=0.0)
STOP = StopMessage(type="stop")
SLEEP = SleepMessage(type="sleep")
IDLE_SLEEP_S = 120.0


@pytest.fixture
def session(motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig) -> TeleopSession:
    return TeleopSession(DriveController(motors, clock, drive_config), clock=clock)


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
    session = TeleopSession(DriveController(motors, clock, drive_config), clock=clock)
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
    return TeleopSession(DriveController(motors, clock, drive_config), gripper, clock=clock)


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


def jog(joint: JointName, direction: int) -> ArmJogMessage:
    return ArmJogMessage.model_validate({"type": "arm", "joint": joint, "direction": direction})


@pytest.fixture
def arm_bus() -> FakeServoBus:
    return FakeServoBus(servos={1: FakeServo(position=575), 2: FakeServo(position=870)})


ARM_JOINTS = {
    JointName.BASE: JointCalibration(
        servo_id=1, min_position=164, max_position=940, home_position=575
    ),
    JointName.SHOULDER: JointCalibration(
        servo_id=2, min_position=223, max_position=948, home_position=870
    ),
}


def make_arm_session(
    motors: FakeMotorDriver,
    clock: FakeClock,
    drive_config: DriveConfig,
    arm_bus: FakeServoBus,
    idle_sleep_s: float = 0.0,
) -> TeleopSession:
    arm = ArmController(arm_bus, ARM_JOINTS, speed_cps=150.0)
    drive = DriveController(motors, clock, drive_config)
    return TeleopSession(drive, arm=arm, jog_step=15, clock=clock, idle_sleep_s=idle_sleep_s)


@pytest.fixture
def arm_session(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig, arm_bus: FakeServoBus
) -> TeleopSession:
    return make_arm_session(motors, clock, drive_config, arm_bus)


def test_driver_jogs_arm_joints_one_step_per_message(
    arm_session: TeleopSession, arm_bus: FakeServoBus
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.BASE, -1))
    status = arm_session.handle("a", jog(JointName.BASE, -1))
    assert status.error is None
    assert arm_bus.servos[1].position == 545
    status = arm_session.handle("a", jog(JointName.SHOULDER, 1))
    assert arm_bus.servos[2].position == 885
    assert status.arm == ArmReport(targets={JointName.BASE: 545, JointName.SHOULDER: 885})


def test_observer_cannot_jog_the_arm(arm_session: TeleopSession, arm_bus: FakeServoBus) -> None:
    arm_session.connect("a")
    arm_session.connect("b")
    status = arm_session.handle("b", jog(JointName.BASE, 1))
    assert status.error == "another client is driving"
    assert arm_bus.moves == []


def test_driver_disconnect_leaves_the_arm_holding(
    arm_session: TeleopSession, arm_bus: FakeServoBus, motors: FakeMotorDriver
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_session.disconnect("a")
    assert motors.current.is_stopped
    assert arm_bus.servos[2].torque_enabled  # going limp would drop it


def test_stop_does_not_drop_the_arm(arm_session: TeleopSession, arm_bus: FakeServoBus) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_session.handle("a", STOP)
    assert arm_bus.servos[2].torque_enabled


def test_jogging_an_uncalibrated_joint_is_reported(arm_session: TeleopSession) -> None:
    arm_session.connect("a")
    status = arm_session.handle("a", jog(JointName.ELBOW, 1))
    assert status.error == "arm joint 'elbow' is not calibrated"


def test_arm_failure_is_reported_and_driving_continues(
    arm_session: TeleopSession, arm_bus: FakeServoBus, motors: FakeMotorDriver
) -> None:
    arm_session.connect("a")
    arm_bus.fail_with = ServoError("servo 1 does not answer")
    status = arm_session.handle("a", jog(JointName.BASE, 1))
    assert status.error == "arm failed: servo 1 does not answer"
    assert arm_session.handle("a", FORWARD).moving
    assert not motors.current.is_stopped


def test_without_arm_status_has_none_and_jogs_are_refused(session: TeleopSession) -> None:
    session.connect("a")
    status = session.handle("a", jog(JointName.BASE, 1))
    assert status.arm is None
    assert status.error == "this robot has no arm"


def test_jog_step_must_be_positive(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig
) -> None:
    with pytest.raises(ValueError, match="jog step"):
        TeleopSession(DriveController(motors, clock, drive_config), jog_step=0, clock=clock)


# --- sleep ---------------------------------------------------------------------------------

#: Time the shoulder needs to move home after one jog step up (15 counts at 150 counts/s).
ONE_STEP_HOME_S = 15 / 150.0


@pytest.fixture
def idle_arm_session(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig, arm_bus: FakeServoBus
) -> TeleopSession:
    return make_arm_session(motors, clock, drive_config, arm_bus, idle_sleep_s=IDLE_SLEEP_S)


def test_sleep_moves_the_arm_home_and_then_lets_it_go_limp(
    arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    status = arm_session.handle("a", SLEEP)
    assert status.error is None
    assert status.asleep
    assert arm_bus.servos[2].position == 870
    assert arm_bus.servos[2].torque_enabled  # still holding on the way home
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S - 0.01)
    arm_session.tick()
    assert arm_bus.servos[2].torque_enabled
    clock.advance(0.02)
    arm_session.tick()
    assert not arm_bus.servos[2].torque_enabled
    status = arm_session.status("a")
    assert status.arm == ArmReport()  # every joint limp
    assert status.asleep


def test_sleep_stops_the_tracks(arm_session: TeleopSession, motors: FakeMotorDriver) -> None:
    arm_session.connect("a")
    arm_session.handle("a", FORWARD)
    assert not arm_session.handle("a", SLEEP).moving
    assert motors.current.is_stopped


def test_sleep_leaves_limp_joints_alone(
    arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    arm_session.connect("a")
    assert arm_session.handle("a", SLEEP).asleep
    clock.advance(10.0)
    arm_session.tick()
    assert arm_bus.moves == []
    assert arm_bus.torque_log == []


def test_observer_cannot_put_the_robot_to_sleep(
    arm_session: TeleopSession, arm_bus: FakeServoBus
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_session.connect("b")
    status = arm_session.handle("b", SLEEP)
    assert status.error == "another client is driving"
    assert not status.asleep
    assert arm_bus.servos[2].position == 855


def test_a_jog_wakes_the_arm_before_it_goes_limp(
    arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_session.handle("a", SLEEP)
    status = arm_session.handle("a", jog(JointName.BASE, 1))
    assert not status.asleep
    clock.advance(10.0)
    arm_session.tick()
    assert arm_bus.servos[2].torque_enabled
    assert arm_bus.servos[1].torque_enabled


def test_failed_homing_is_reported_and_the_arm_keeps_holding(
    arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    class DeafToReads(FakeServoBus):
        deaf = False

        def read_state(self, servo_id: int) -> ServoState:
            if self.deaf:
                raise ServoError(f"servo {servo_id} did not answer")
            return super().read_state(servo_id)

    bus = DeafToReads(servos=arm_bus.servos)
    session = make_arm_session(FakeMotorDriver(), clock, DriveConfig(), bus)
    session.connect("a")
    session.handle("a", jog(JointName.SHOULDER, -1))
    bus.deaf = True
    status = session.handle("a", SLEEP)
    assert status.error == "arm failed: servo 2 did not answer"
    assert bus.servos[2].torque_enabled  # going limp away from home could drop it
    clock.advance(10.0)
    session.tick()
    assert bus.servos[2].torque_enabled


def test_idle_arm_falls_asleep(
    idle_arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    idle_arm_session.connect("a")
    idle_arm_session.handle("a", jog(JointName.SHOULDER, -1))
    clock.advance(IDLE_SLEEP_S - 0.1)
    idle_arm_session.tick()
    assert not idle_arm_session.status("a").asleep
    assert arm_bus.servos[2].position == 855
    clock.advance(0.1)
    idle_arm_session.tick()
    assert idle_arm_session.status("a").asleep
    assert arm_bus.servos[2].position == 870
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S + 0.01)
    idle_arm_session.tick()
    assert not arm_bus.servos[2].torque_enabled


def test_idle_sleep_also_relaxes_the_arm_after_the_driver_left(
    idle_arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    idle_arm_session.connect("a")
    idle_arm_session.handle("a", jog(JointName.SHOULDER, -1))
    idle_arm_session.disconnect("a")
    assert arm_bus.servos[2].torque_enabled
    clock.advance(IDLE_SLEEP_S)
    idle_arm_session.tick()
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S + 0.01)
    idle_arm_session.tick()
    assert not arm_bus.servos[2].torque_enabled
    assert arm_bus.servos[2].position == 870


def test_idle_sleep_does_not_stop_the_tracks(
    idle_arm_session: TeleopSession, motors: FakeMotorDriver, clock: FakeClock
) -> None:
    idle_arm_session.connect("a")
    idle_arm_session.handle("a", jog(JointName.SHOULDER, -1))
    clock.advance(IDLE_SLEEP_S)
    idle_arm_session.handle("a", FORWARD)
    assert not idle_arm_session.tick()
    assert idle_arm_session.status("a").asleep
    assert not motors.current.is_stopped


def test_a_limp_robot_is_never_put_to_sleep_by_the_idle_timer(
    idle_arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    idle_arm_session.connect("a")
    clock.advance(10 * IDLE_SLEEP_S)
    idle_arm_session.tick()
    assert not idle_arm_session.status("a").asleep
    assert arm_bus.moves == []


def test_idle_sleep_retries_after_a_servo_failure(
    idle_arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    idle_arm_session.connect("a")
    idle_arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_bus.fail_with = ServoError("servo 2 does not answer")
    clock.advance(IDLE_SLEEP_S)
    idle_arm_session.tick()  # logged, not raised: the tracks must keep working
    arm_bus.fail_with = None
    clock.advance(1.0)
    idle_arm_session.tick()
    assert arm_bus.servos[2].torque_enabled  # no retry before another idle period
    clock.advance(IDLE_SLEEP_S)
    idle_arm_session.tick()
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S + 0.01)
    idle_arm_session.tick()
    assert not arm_bus.servos[2].torque_enabled


def test_sleep_releases_the_gripper(
    gripper_session: TeleopSession, servo_bus: FakeServoBus
) -> None:
    gripper_session.connect("a")
    gripper_session.handle("a", gripper_message("close"))
    status = gripper_session.handle("a", SLEEP)
    assert status.gripper == GripperReport(engaged=False)
    assert not servo_bus.servos[4].torque_enabled


def test_a_gripper_command_wakes_the_robot(gripper_session: TeleopSession) -> None:
    gripper_session.connect("a")
    gripper_session.handle("a", SLEEP)
    assert not gripper_session.handle("a", gripper_message("open")).asleep


def test_idle_gripper_falls_asleep(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig, servo_bus: FakeServoBus
) -> None:
    calibration = GripperCalibration(servo_id=4, open_position=745, closed_position=965)
    gripper = GripperController(servo_bus, calibration, move_time_s=0.8, max_torque=0.5)
    drive = DriveController(motors, clock, drive_config)
    session = TeleopSession(drive, gripper, clock=clock, idle_sleep_s=IDLE_SLEEP_S)
    session.connect("a")
    session.handle("a", gripper_message("close"))
    clock.advance(IDLE_SLEEP_S)
    session.tick()
    assert not servo_bus.servos[4].torque_enabled
    assert session.status("a").gripper == GripperReport(engaged=False)


def test_idle_sleep_time_must_not_be_negative(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig
) -> None:
    with pytest.raises(ValueError, match="idle sleep"):
        TeleopSession(DriveController(motors, clock, drive_config), clock=clock, idle_sleep_s=-1)


def test_a_failed_release_is_logged_and_does_not_stop_teleop(
    arm_session: TeleopSession, arm_bus: FakeServoBus, clock: FakeClock
) -> None:
    arm_session.connect("a")
    arm_session.handle("a", jog(JointName.SHOULDER, -1))
    arm_session.handle("a", SLEEP)
    arm_bus.fail_with = ServoError("servo 2 does not answer")
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S + 0.01)
    assert not arm_session.tick()  # no exception: the tracks must keep working
    assert arm_session.handle("a", FORWARD).moving
