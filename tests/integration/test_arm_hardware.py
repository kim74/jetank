"""Bring-up test for the arm joints (base, shoulder, elbow, camera tilt) under power.

Read-only check (nothing moves, no torque is switched on):

    pytest -m hardware tests/integration/test_arm_hardware.py

The motion check holds every joint, moves all of them home, then drives one joint at a time
to 20 counts inside each of its configured limits and back home, the others holding home.
Keep hands, cables and objects clear of the arm's reach, then:

    JETANK_ALLOW_MOTION=1 pytest -m hardware tests/integration/test_arm_hardware.py -s

Whatever happens, the arm is sent home and then goes limp (torque off); from home it sags
only a little. The gripper is not touched. Settings come from ``config/jetank.yaml`` if it
exists; the servo backend is forced to ``hardware``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from jetank.config.settings import load_settings
from jetank.control.arm import ArmController
from jetank.domain.servo import JointName
from jetank.drivers.scs_bus import ScsServoBus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jetank.config.settings import JetankSettings
    from jetank.domain.servo import ServoState

pytestmark = pytest.mark.hardware

logger = logging.getLogger(__name__)

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "jetank.yaml"
#: Lightest joint first, the shoulder (which carries the whole arm) last.
TEST_ORDER: Final = (JointName.CAMERA_TILT, JointName.BASE, JointName.ELBOW, JointName.SHOULDER)
#: Targets stay this many counts inside the configured limits.
LIMIT_MARGIN: Final = 20
#: How close a joint must settle to its target, in counts (~6 degrees). The servos hold with a
#: load-dependent offset: the shoulder, carrying the whole arm, settles ~13 counts below home.
#: A real stall (blocked joint, sensor blind zone) is off by hundreds.
POSITION_TOLERANCE: Final = 25
#: A joint counts as settled when its position changes by at most 1 count over this time.
STILL_S: Final = 0.1
#: Extra time allowed beyond the planned move duration.
SETTLE_S: Final = 1.5
POLL_S: Final = 0.02

needs_motion = pytest.mark.skipif(
    os.environ.get("JETANK_ALLOW_MOTION") != "1",
    reason="set JETANK_ALLOW_MOTION=1 to allow the arm to move",
)


def _settings() -> JetankSettings:
    settings = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
    servos = settings.servos.model_copy(update={"backend": "hardware"})
    return settings.model_copy(update={"servos": servos})


@pytest.fixture
def arm() -> Iterator[ArmController]:
    """Arm controller; if a test engaged it, teardown sends it home and then goes limp."""
    settings = _settings()
    with (
        ScsServoBus.open(settings.servos) as bus,
        ArmController(bus, settings.arm.joints, settings.arm.speed_cps) as controller,
    ):
        try:
            yield controller
        finally:
            try:
                if controller.engaged:
                    time.sleep(controller.home() + SETTLE_S)
            except Exception:  # still go limp below, whatever failed while homing
                logger.exception("could not send the arm home")


def _settle(arm: ArmController, joint: JointName, duration_s: float) -> tuple[ServoState, float]:
    """Wait out the planned move, then until the joint stops moving or time runs out."""
    start = time.monotonic()
    time.sleep(duration_s)
    previous = arm.state(joint)
    while True:
        time.sleep(STILL_S)
        state = arm.state(joint)
        elapsed_s = time.monotonic() - start
        still = not state.moving and abs(state.position - previous.position) <= 1
        if still or elapsed_s > duration_s + SETTLE_S:
            return state, elapsed_s
        previous = state


def _drive(arm: ArmController, joint: JointName, target: int) -> ServoState:
    duration_s = arm.move(joint, target)
    state, elapsed_s = _settle(arm, joint, duration_s)
    print(
        f"{joint.value:>11}: target {target:4d}, settled {state.position:4d} "
        f"(error {state.position - target:+3d}) after {elapsed_s:4.2f} s, load {state.load:+.2f}"
    )
    assert abs(state.position - target) <= POSITION_TOLERANCE, f"{joint.value} stalled"
    assert state.torque_enabled
    assert state.error_flags == 0
    return state


def test_every_arm_joint_answers(arm: ArmController) -> None:
    print()
    for joint in TEST_ORDER:
        state = arm.state(joint)
        print(
            f"{joint.value:>11}: {state.position:4d}, {state.voltage_v} V, "
            f"torque {state.torque_enabled}"
        )
        assert state.error_flags == 0


@needs_motion
def test_each_joint_reaches_both_limits_under_power(arm: ArmController) -> None:
    print()
    arm.hold()
    time.sleep(arm.home() + SETTLE_S)
    for joint in TEST_ORDER:
        _drive(arm, joint, arm.joints[joint].home_position)
    for joint in TEST_ORDER:
        cal = arm.joints[joint]
        for target in (
            cal.min_position + LIMIT_MARGIN,
            cal.max_position - LIMIT_MARGIN,
            cal.home_position,
        ):
            _drive(arm, joint, target)
