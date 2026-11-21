from __future__ import annotations

import pytest

from jetank.control.arm import PARK_SETTLE_S, ArmController
from jetank.control.servo_sleep import ServoSleep
from jetank.domain.servo import JointCalibration, JointName
from jetank.drivers.fake import FakeClock, FakeServo, FakeServoBus
from jetank.errors import ServoError

IDLE_S = 120.0
SHOULDER = {
    JointName.SHOULDER: JointCalibration(
        servo_id=2, min_position=223, max_position=948, home_position=870
    )
}
ONE_STEP_HOME_S = 15 / 150.0


@pytest.fixture
def bus() -> FakeServoBus:
    return FakeServoBus(servos={2: FakeServo(position=870)})


@pytest.fixture
def arm(bus: FakeServoBus) -> ArmController:
    return ArmController(bus, SHOULDER, speed_cps=150.0)


def engaged_off_home(arm: ArmController) -> None:
    arm.jog(JointName.SHOULDER, -15)


def test_fall_asleep_homes_the_arm_and_releases_it_once_there(
    arm: ArmController, bus: FakeServoBus, clock: FakeClock
) -> None:
    sleep = ServoSleep(arm, clock, IDLE_S)
    engaged_off_home(arm)
    assert sleep.fall_asleep() is None
    assert sleep.asleep
    assert bus.servos[2].position == 870
    clock.advance(ONE_STEP_HOME_S + PARK_SETTLE_S - 0.01)
    assert not sleep.tick(servos_powered=True)
    assert bus.servos[2].torque_enabled
    clock.advance(0.02)
    assert not sleep.tick(servos_powered=True)
    assert not bus.servos[2].torque_enabled


def test_wake_cancels_a_pending_release(
    arm: ArmController, bus: FakeServoBus, clock: FakeClock
) -> None:
    sleep = ServoSleep(arm, clock, IDLE_S)
    engaged_off_home(arm)
    sleep.fall_asleep()
    sleep.wake()
    assert not sleep.asleep
    clock.advance(10.0)
    sleep.tick(servos_powered=True)
    assert bus.servos[2].torque_enabled


def test_idle_timer_asks_for_sleep_only_while_a_servo_is_powered(clock: FakeClock) -> None:
    sleep = ServoSleep(None, clock, IDLE_S)
    clock.advance(IDLE_S - 0.1)
    assert not sleep.tick(servos_powered=True)
    clock.advance(0.2)
    assert not sleep.tick(servos_powered=False)
    assert sleep.tick(servos_powered=True)


def test_a_command_restarts_the_idle_timer(clock: FakeClock) -> None:
    sleep = ServoSleep(None, clock, IDLE_S)
    clock.advance(IDLE_S - 1.0)
    sleep.wake()
    clock.advance(2.0)
    assert not sleep.tick(servos_powered=True)


def test_zero_disables_the_idle_timer(clock: FakeClock) -> None:
    sleep = ServoSleep(None, clock, 0.0)
    clock.advance(1e6)
    assert not sleep.tick(servos_powered=True)


def test_failed_homing_keeps_holding_and_retries_after_another_idle_period(
    arm: ArmController, bus: FakeServoBus, clock: FakeClock
) -> None:
    sleep = ServoSleep(arm, clock, IDLE_S)
    engaged_off_home(arm)
    clock.advance(IDLE_S)
    assert sleep.tick(servos_powered=True)
    bus.fail_with = ServoError("servo 2 does not answer")
    error = sleep.fall_asleep()
    assert error is not None
    assert "arm failed" in error
    bus.fail_with = None
    assert bus.servos[2].torque_enabled
    assert not sleep.tick(servos_powered=True)  # timer restarted
    clock.advance(IDLE_S)
    assert sleep.tick(servos_powered=True)


def test_a_limp_arm_is_not_moved(arm: ArmController, bus: FakeServoBus, clock: FakeClock) -> None:
    sleep = ServoSleep(arm, clock, IDLE_S)
    assert sleep.fall_asleep() is None
    assert sleep.asleep
    assert not bus.servos[2].torque_enabled


def test_idle_sleep_time_must_not_be_negative(clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="idle sleep"):
        ServoSleep(None, clock, -1.0)
