from __future__ import annotations

import pytest

from jetank.control.gripper import MIN_MOVE_S, GripperController
from jetank.domain.servo import GripperCalibration
from jetank.drivers.fake import FakeServo, FakeServoBus
from jetank.errors import DriverClosedError, ServoError

CAL = GripperCalibration(servo_id=4, open_position=745, closed_position=965)
MOVE_TIME_S = 0.8


@pytest.fixture
def bus() -> FakeServoBus:
    # Gripper half open; another servo (the arm) on the same bus.
    return FakeServoBus(servos={4: FakeServo(position=855, goal=200), 2: FakeServo(position=906)})


@pytest.fixture
def gripper(bus: FakeServoBus) -> GripperController:
    return GripperController(bus, CAL, MOVE_TIME_S)


def test_open_and_shut_go_to_the_limits(gripper: GripperController, bus: FakeServoBus) -> None:
    assert gripper.open() == 745
    assert bus.servos[4].position == 745
    assert gripper.shut() == 965
    assert bus.servos[4].position == 965


def test_closing_aims_past_the_closed_limit_to_squeeze(bus: FakeServoBus) -> None:
    cal = GripperCalibration(servo_id=4, open_position=955, closed_position=740, grip_overshoot=40)
    gripper = GripperController(bus, cal, MOVE_TIME_S)
    assert gripper.shut() == 700
    assert bus.moves[-1].position == 700
    assert gripper.set_opening(0.0) == 700


def test_any_other_opening_stays_within_the_limits(bus: FakeServoBus) -> None:
    cal = GripperCalibration(servo_id=4, open_position=955, closed_position=740, grip_overshoot=40)
    gripper = GripperController(bus, cal, MOVE_TIME_S)
    assert gripper.set_opening(0.01) == 742
    assert gripper.open() == 955


def test_torque_limit_is_set_before_torque_on(bus: FakeServoBus) -> None:
    order: list[str] = []

    class OrderedBus(FakeServoBus):
        def set_torque_limit(self, servo_id: int, fraction: float) -> None:
            order.append(f"limit {fraction}")
            super().set_torque_limit(servo_id, fraction)

        def set_torque(self, servo_id: int, enabled: bool) -> None:
            order.append(f"torque {enabled}")
            super().set_torque(servo_id, enabled)

    ordered = OrderedBus(servos={4: FakeServo(position=855)})
    GripperController(ordered, CAL, MOVE_TIME_S, max_torque=0.4).shut()
    assert order == ["limit 0.4", "torque True"]


@pytest.mark.parametrize("max_torque", [0.0, 1.5])
def test_rejects_invalid_max_torque(bus: FakeServoBus, max_torque: float) -> None:
    with pytest.raises(ValueError, match="max torque"):
        GripperController(bus, CAL, MOVE_TIME_S, max_torque=max_torque)


def test_goal_is_parked_at_present_position_before_torque_on(
    gripper: GripperController, bus: FakeServoBus
) -> None:
    # The stale goal (200) must never be driven to when torque switches on.
    gripper.set_opening(1.0)
    first = bus.moves[0]
    assert (first.position, first.torque_enabled) == (855, False)
    assert bus.torque_log == [(4, True)]
    assert bus.moves[1].position == 745
    assert bus.moves[1].torque_enabled


def test_torque_is_enabled_only_once(gripper: GripperController, bus: FakeServoBus) -> None:
    gripper.open()
    gripper.shut()
    assert bus.torque_log == [(4, True)]


def test_resting_outside_limits_does_not_jump(bus: FakeServoBus) -> None:
    bus.servos[4].position = 1022
    GripperController(bus, CAL, MOVE_TIME_S).set_opening(0.0)
    assert [m.position for m in bus.moves] == [1022, 965]


def test_move_time_scales_with_distance(gripper: GripperController, bus: FakeServoBus) -> None:
    gripper.open()  # 855 -> 745: half a stroke
    assert bus.moves[-1].duration_s == pytest.approx(MOVE_TIME_S / 2)
    gripper.set_opening(1.0)  # already there: minimum time
    assert bus.moves[-1].duration_s == MIN_MOVE_S


@pytest.mark.parametrize("opening", [-0.1, 1.1])
def test_invalid_opening_sends_nothing(
    gripper: GripperController, bus: FakeServoBus, opening: float
) -> None:
    with pytest.raises(ValueError, match="opening"):
        gripper.set_opening(opening)
    assert bus.moves == []
    assert bus.torque_log == []


def test_only_the_gripper_servo_is_touched(gripper: GripperController, bus: FakeServoBus) -> None:
    gripper.open()
    gripper.close()
    assert {m.servo_id for m in bus.moves} == {4}
    assert {servo_id for servo_id, _ in bus.torque_log} == {4}
    assert bus.servos[2].position == 906


def test_status_is_read_only(gripper: GripperController, bus: FakeServoBus) -> None:
    status = gripper.status()
    gripper.close()
    assert status.opening == pytest.approx(0.5)
    assert status.servo.position == 855
    assert bus.moves == []
    assert bus.torque_log == []


def test_close_releases_after_a_move(gripper: GripperController, bus: FakeServoBus) -> None:
    with gripper:
        gripper.open()
    assert bus.torque_log[-1] == (4, False)
    assert not bus.servos[4].torque_enabled
    gripper.close()  # idempotent
    with pytest.raises(DriverClosedError):
        gripper.open()


def test_release_lets_go_and_next_move_reengages(
    gripper: GripperController, bus: FakeServoBus
) -> None:
    gripper.open()
    gripper.release()
    gripper.shut()
    assert bus.torque_log == [(4, True), (4, False), (4, True)]


def test_failed_torque_enable_is_still_switched_off_on_close(bus: FakeServoBus) -> None:
    class LossyBus(FakeServoBus):
        def set_torque(self, servo_id: int, enabled: bool) -> None:
            super().set_torque(servo_id, enabled)
            if enabled:
                raise ServoError("acknowledgement lost")

    lossy = LossyBus(servos={4: FakeServo(position=855)})
    gripper = GripperController(lossy, CAL, MOVE_TIME_S)
    with pytest.raises(ServoError):
        gripper.open()
    gripper.close()
    assert not lossy.servos[4].torque_enabled


def test_rejects_non_positive_move_time(bus: FakeServoBus) -> None:
    with pytest.raises(ValueError, match="positive"):
        GripperController(bus, CAL, 0.0)


def test_failed_park_is_still_switched_off_on_close() -> None:
    class LossyBus(FakeServoBus):
        def move(self, servo_id: int, position: int, duration_s: float) -> None:
            super().move(servo_id, position, duration_s)
            raise ServoError("acknowledgement lost")

    lossy = LossyBus(servos={4: FakeServo(position=855)})
    gripper = GripperController(lossy, CAL, MOVE_TIME_S)
    with pytest.raises(ServoError):
        gripper.open()
    gripper.close()
    assert lossy.torque_log == [(4, False)]
