"""Bring-up test for the gripper servo on the JETANK's serial bus.

Read-only checks (nothing moves, no torque is switched on):

    pytest -m hardware tests/integration/test_gripper_hardware.py

Motion checks open and close the gripper slowly within the configured limits. Keep fingers
and objects out of the jaws, then:

    JETANK_ALLOW_MOTION=1 pytest -m hardware tests/integration/test_gripper_hardware.py -s

Whatever happens, the gripper's torque is switched off at the end (it goes limp).
Settings come from ``config/jetank.yaml`` if it exists; the servo backend is forced to
``hardware``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from jetank.config.settings import load_settings
from jetank.control.gripper import GripperController
from jetank.drivers.scs_bus import ScsServoBus

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jetank.config.servos import GripperConfig, ServosConfig
    from jetank.domain.servo import GripperStatus

pytestmark = pytest.mark.hardware

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "jetank.yaml"
JETANK_SERVO_IDS: Final = (1, 2, 3, 4, 5)
#: Servo supply range seen on the JETANK (~5.4 V) with headroom; SCS servos run 4.5-8.4 V.
PLAUSIBLE_SUPPLY_V: Final = (4.5, 8.4)
#: How close the servo must get to its target, in counts (~3 degrees).
POSITION_TOLERANCE: Final = 15
POLL_S: Final = 0.02

needs_motion = pytest.mark.skipif(
    os.environ.get("JETANK_ALLOW_MOTION") != "1",
    reason="set JETANK_ALLOW_MOTION=1 to allow the gripper to move",
)


def _configs() -> tuple[ServosConfig, GripperConfig]:
    settings = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
    return settings.servos.model_copy(update={"backend": "hardware"}), settings.gripper


@pytest.fixture
def bus() -> Iterator[ScsServoBus]:
    with ScsServoBus.open(_configs()[0]) as servo_bus:
        yield servo_bus


@pytest.fixture
def gripper(bus: ScsServoBus) -> Iterator[GripperController]:
    """Gripper controller; its teardown switches the torque off even if a test failed."""
    config = _configs()[1]
    with GripperController(
        bus, config.calibration, config.move_time_s, config.max_torque
    ) as controller:
        yield controller


def _wait_until_near(
    gripper: GripperController, target: int, timeout_s: float
) -> tuple[GripperStatus, float]:
    start = time.monotonic()
    while True:
        status = gripper.status()
        elapsed = time.monotonic() - start
        if abs(status.servo.position - target) <= POSITION_TOLERANCE or elapsed > timeout_s:
            return status, elapsed
        time.sleep(POLL_S)


def test_all_jetank_servos_answer(bus: ScsServoBus) -> None:
    missing = [servo_id for servo_id in JETANK_SERVO_IDS if not bus.ping(servo_id)]
    assert missing == []


def test_gripper_status_is_plausible(gripper: GripperController) -> None:
    status = gripper.status()
    low, high = PLAUSIBLE_SUPPLY_V
    assert low <= status.servo.voltage_v <= high
    assert status.servo.error_flags == 0
    print(f"\ngripper at rest: {status.servo}")


@needs_motion
def test_gripper_opens_and_closes_slowly_within_limits(gripper: GripperController) -> None:
    config = _configs()[1]
    for opening in (0.5, 1.0, 0.0, 0.5):
        target = gripper.set_opening(opening)
        status, elapsed_s = _wait_until_near(gripper, target, config.move_time_s * 3)
        reached = status.servo.position
        print(f"\nopening {opening}: target {target}, reached {reached} in {elapsed_s:.2f} s")
        assert abs(status.servo.position - target) <= POSITION_TOLERANCE
        assert status.servo.torque_enabled
        assert status.servo.error_flags == 0


@needs_motion
def test_torque_limit_is_applied_on_the_servo(bus: ScsServoBus) -> None:
    config = _configs()[1]
    with GripperController(
        bus, config.calibration, config.move_time_s, config.max_torque
    ) as gripper:
        gripper.set_opening(0.5)
        assert bus.read_torque_limit(config.servo_id) == pytest.approx(config.max_torque)


@needs_motion
def test_gripper_is_limp_after_close(bus: ScsServoBus) -> None:
    config = _configs()[1]
    with GripperController(
        bus, config.calibration, config.move_time_s, config.max_torque
    ) as gripper:
        gripper.set_opening(0.5)
        _wait_until_near(gripper, config.calibration.target(0.5), config.move_time_s * 3)
    assert not bus.read_state(config.servo_id).torque_enabled
