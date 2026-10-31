from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jetank.domain.servo import GripperCalibration, ServoState

# JETANK: opening lowers the position number.
CAL = GripperCalibration(servo_id=4, open_position=745, closed_position=965)


def test_target_maps_closed_and_open_to_the_limits() -> None:
    assert CAL.target(0.0) == 965
    assert CAL.target(1.0) == 745
    assert CAL.target(0.5) == 855


def test_target_works_in_either_direction() -> None:
    rising = GripperCalibration(servo_id=1, open_position=900, closed_position=500)
    assert rising.target(0.25) == 600


@pytest.mark.parametrize("opening", [-0.01, 1.01, float("nan")])
def test_target_rejects_invalid_opening(opening: float) -> None:
    with pytest.raises(ValueError, match="opening"):
        CAL.target(opening)


@given(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0))
def test_target_stays_within_limits_and_is_monotonic(a: float, b: float) -> None:
    low, high = sorted((a, b))
    assert 745 <= CAL.target(low) <= 965
    assert CAL.target(high) <= CAL.target(low)  # more open = lower number here


def test_opening_is_inverse_of_target_and_clamped() -> None:
    assert CAL.opening(855) == pytest.approx(0.5)
    assert CAL.opening(1022) == 0.0
    assert CAL.opening(700) == 1.0


def test_clamp_pulls_positions_into_the_limits() -> None:
    assert CAL.clamp(1022) == 965
    assert CAL.clamp(700) == 745
    assert CAL.clamp(800) == 800


def test_calibration_needs_enough_travel() -> None:
    with pytest.raises(ValidationError, match="20 counts apart"):
        GripperCalibration(servo_id=4, open_position=900, closed_position=910)


def test_calibration_rejects_positions_beyond_servo_range() -> None:
    with pytest.raises(ValidationError):
        GripperCalibration(servo_id=4, open_position=745, closed_position=1024)


def test_servo_state_rejects_impossible_readings() -> None:
    with pytest.raises(ValidationError):
        ServoState(
            servo_id=4,
            position=2000,
            voltage_v=5.0,
            temperature_c=25,
            moving=False,
            torque_enabled=False,
        )


def test_joint_limits_must_be_ordered_and_contain_home() -> None:
    from jetank.domain.servo import JointCalibration

    with pytest.raises(ValidationError, match="below"):
        JointCalibration(servo_id=1, min_position=900, max_position=500, home_position=700)
    with pytest.raises(ValidationError, match="within the limits"):
        JointCalibration(servo_id=1, min_position=500, max_position=900, home_position=950)


def test_joint_clamp() -> None:
    from jetank.domain.servo import JointCalibration

    joint = JointCalibration(servo_id=1, min_position=460, max_position=983, home_position=875)
    assert (joint.clamp(1022), joint.clamp(300), joint.clamp(700)) == (983, 460, 700)
