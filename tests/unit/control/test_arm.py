from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.control.arm import MAX_MOVE_S, MIN_MOVE_S, ArmController
from jetank.domain.servo import JointCalibration, JointName
from jetank.drivers.fake import FakeServo, FakeServoBus
from jetank.errors import ConfigurationError, DriverClosedError, ServoError

JOINTS = {
    JointName.SHOULDER: JointCalibration(
        servo_id=2, min_position=223, max_position=948, home_position=870
    ),
    JointName.ELBOW: JointCalibration(
        servo_id=3, min_position=90, max_position=985, home_position=479
    ),
}
SPEED_CPS = 100.0


@pytest.fixture
def bus() -> FakeServoBus:
    # Limp arm; the elbow rests outside its limits with a stale goal. The gripper shares the bus.
    return FakeServoBus(
        servos={
            2: FakeServo(position=880, goal=300),
            3: FakeServo(position=1010, goal=100),
            4: FakeServo(position=800),
        }
    )


@pytest.fixture
def arm(bus: FakeServoBus) -> ArmController:
    return ArmController(bus, JOINTS, SPEED_CPS)


def test_move_reaches_target_within_limits(arm: ArmController, bus: FakeServoBus) -> None:
    arm.move(JointName.SHOULDER, 500)
    assert bus.servos[2].position == 500
    assert bus.servos[2].torque_enabled


def test_targets_beyond_the_limits_are_clamped(arm: ArmController, bus: FakeServoBus) -> None:
    arm.move(JointName.ELBOW, 1023)
    assert bus.servos[3].position == 985
    arm.move(JointName.ELBOW, 0)
    assert bus.servos[3].position == 90


def test_engaging_parks_the_goal_so_torque_on_does_not_jump(
    arm: ArmController, bus: FakeServoBus
) -> None:
    arm.hold()
    assert bus.servos[2].position == 880
    assert bus.servos[3].position == 1010  # held where it rests, even outside the limits
    assert bus.moves[0].position == 880
    assert not bus.moves[0].torque_enabled


def test_move_duration_follows_the_speed(arm: ArmController, bus: FakeServoBus) -> None:
    assert arm.move(JointName.SHOULDER, 680) == pytest.approx(200 / SPEED_CPS)
    assert bus.moves[-1].duration_s == pytest.approx(2.0)


@given(start=st.integers(0, 1023), target=st.integers(-100, 1200))
def test_move_duration_stays_within_the_servo_range(start: int, target: int) -> None:
    bus = FakeServoBus(servos={3: FakeServo(position=start)})
    arm = ArmController(bus, {JointName.ELBOW: JOINTS[JointName.ELBOW]}, 1.0)
    duration_s = arm.move(JointName.ELBOW, target)
    assert MIN_MOVE_S <= duration_s <= MAX_MOVE_S
    assert 90 <= bus.moves[-1].position <= 985


def test_home_moves_every_joint_home(arm: ArmController, bus: FakeServoBus) -> None:
    longest_s = arm.home()
    assert bus.servos[2].position == 870
    assert bus.servos[3].position == 479
    assert longest_s == pytest.approx((1010 - 479) / SPEED_CPS)


def test_uncalibrated_joint_cannot_be_moved(arm: ArmController) -> None:
    with pytest.raises(ConfigurationError, match="base"):
        arm.move(JointName.BASE, 800)


def test_only_the_arm_servos_are_touched(arm: ArmController, bus: FakeServoBus) -> None:
    arm.hold()
    arm.home()
    arm.close()
    touched = {move.servo_id for move in bus.moves} | {sid for sid, _ in bus.torque_log}
    assert touched == {2, 3}


def test_close_releases_every_engaged_joint(arm: ArmController, bus: FakeServoBus) -> None:
    arm.move(JointName.ELBOW, 500)
    assert arm.engaged
    arm.close()
    assert bus.torque_log[-1] == (3, False)
    assert not bus.servos[3].torque_enabled
    assert (2, False) not in bus.torque_log  # never engaged, so nothing sent


def test_close_without_motion_sends_nothing(arm: ArmController, bus: FakeServoBus) -> None:
    arm.state(JointName.SHOULDER)
    assert not arm.engaged
    arm.close()
    assert bus.torque_log == []


def test_release_tries_every_joint_and_raises_the_first_failure(bus: FakeServoBus) -> None:
    class FlakyBus(FakeServoBus):
        def set_torque(self, servo_id: int, enabled: bool) -> None:
            if servo_id == 3 and not enabled:
                raise ServoError("no reply")
            super().set_torque(servo_id, enabled)

    flaky = FlakyBus(servos=bus.servos)
    arm = ArmController(flaky, JOINTS, SPEED_CPS)
    arm.hold()
    with pytest.raises(ServoError):
        arm.release()
    assert not flaky.servos[2].torque_enabled


def test_closed_controller_refuses_commands(arm: ArmController) -> None:
    with arm:
        pass
    arm.close()  # idempotent
    with pytest.raises(DriverClosedError):
        arm.move(JointName.SHOULDER, 500)


def test_speed_must_be_positive(bus: FakeServoBus) -> None:
    with pytest.raises(ValueError, match="speed"):
        ArmController(bus, JOINTS, 0.0)
