from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.domain.kinematics import mix
from jetank.domain.models import DriveCommand, TrackSpeeds

unit = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)


@pytest.mark.parametrize(
    ("linear", "angular", "expected"),
    [
        (0.0, 0.0, (0.0, 0.0)),
        (0.6, 0.0, (0.6, 0.6)),
        (-0.6, 0.0, (-0.6, -0.6)),
        (0.0, 0.5, (-0.5, 0.5)),  # positive angular = turn left: right track faster
        (0.5, 0.25, (0.25, 0.75)),
    ],
)
def test_mix_within_range(linear: float, angular: float, expected: tuple[float, float]) -> None:
    speeds = mix(DriveCommand(linear=linear, angular=angular))
    assert (speeds.left, speeds.right) == pytest.approx(expected)


def test_mix_saturated_command_keeps_track_ratio() -> None:
    speeds = mix(DriveCommand(linear=1.0, angular=0.5))
    # Unsaturated would be (0.5, 1.5); scaled by 1/1.5.
    assert (speeds.left, speeds.right) == pytest.approx((1 / 3, 1.0))


@given(linear=unit, angular=unit)
def test_mix_never_exceeds_full_scale(linear: float, angular: float) -> None:
    speeds = mix(DriveCommand(linear=linear, angular=angular))
    assert -1.0 <= speeds.left <= 1.0
    assert -1.0 <= speeds.right <= 1.0


@given(linear=unit, angular=unit)
def test_mix_mirrors_when_turn_is_reversed(linear: float, angular: float) -> None:
    speeds = mix(DriveCommand(linear=linear, angular=angular))
    mirrored = mix(DriveCommand(linear=linear, angular=-angular))
    assert mirrored == TrackSpeeds(left=speeds.right, right=speeds.left)


@given(linear=unit, angular=unit)
def test_mix_never_turns_the_wrong_way(linear: float, angular: float) -> None:
    speeds = mix(DriveCommand(linear=linear, angular=angular))
    # Tiny angular values may round away entirely, but must never flip the turn direction.
    assert (speeds.right - speeds.left) * angular >= 0
