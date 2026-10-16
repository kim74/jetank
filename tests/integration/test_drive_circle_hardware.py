"""Ground test: the JETANK drives a slow counter-clockwise circle while the operator watches.

Put the robot on the ground with at least 1.5 m of clear space around it, then run:

    JETANK_ALLOW_GROUND_DRIVE=1 pytest -m hardware -s \
        tests/integration/test_drive_circle_hardware.py

Ctrl-C stops the robot at any time. The run is bounded by ``JETANK_CIRCLE_S`` seconds
(default 8, at most 30); commands go through the :class:`DriveController`, so its watchdog
also stops the tracks if the loop ever stalls. Without ``-s`` nobody can confirm the robot is
on safe ground, so the test is skipped.

Settings come from ``config/jetank.yaml`` if it exists, otherwise the defaults, and
``JETANK_*`` environment variables override both.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Final

import pytest

from jetank.config.settings import DriveConfig, MotorsConfig, load_settings
from jetank.control.drive import DriveController
from jetank.domain.models import DriveCommand
from jetank.drivers.system_clock import SystemClock
from jetank.drivers.tb6612_tracks import Tb6612TrackDriver

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("JETANK_ALLOW_GROUND_DRIVE") != "1",
        reason="set JETANK_ALLOW_GROUND_DRIVE=1 to let the robot drive on the ground",
    ),
]

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "jetank.yaml"
# left = 0.75 - 0.25 = 0.5, right = 1.0; scaled by MAX_SPEED to 25 % and 50 % motor power.
CIRCLE: Final = DriveCommand(linear=0.75, angular=0.25)
MAX_SPEED: Final = 0.5
WATCHDOG_TIMEOUT_S: Final = 0.3
COMMAND_PERIOD_S: Final = 0.05
DEFAULT_DURATION_S: Final = 8.0
MAX_DURATION_S: Final = 30.0


def _duration_s() -> float:
    duration_s = float(os.environ.get("JETANK_CIRCLE_S", DEFAULT_DURATION_S))
    if not 0.0 < duration_s <= MAX_DURATION_S:
        pytest.fail(f"JETANK_CIRCLE_S must be within (0, {MAX_DURATION_S}], got {duration_s}")
    return duration_s


def _ask(question: str) -> bool:
    return input(f"\n{question} [y/n] ").strip().lower() == "y"


@pytest.fixture
def config(pytestconfig: pytest.Config) -> MotorsConfig:
    if pytestconfig.getoption("capture") != "no":
        pytest.skip("run with -s: the operator must confirm the robot is on safe ground")
    if not _ask("Is the robot on the ground with at least 1.5 m clear around it?"):
        pytest.skip("ground not clear; circle drive not run")
    return load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None).motors


def test_drive_counter_clockwise_circle(config: MotorsConfig) -> None:
    duration_s = _duration_s()
    clock = SystemClock()
    drive_config = DriveConfig(max_speed=MAX_SPEED, watchdog_timeout_s=WATCHDOG_TIMEOUT_S)

    # Both context managers stop the tracks on exit, including on Ctrl-C or a failed assert.
    with (
        Tb6612TrackDriver.open(config) as driver,
        DriveController(driver, clock, drive_config) as controller,
    ):
        print(f"\nDriving a counter-clockwise circle for {duration_s:.0f} s (Ctrl-C stops)")
        sent = controller.command(CIRCLE)
        assert 0.0 < sent.left < sent.right, "outer (right) track must run faster"
        end_s = clock.monotonic() + duration_s
        while clock.monotonic() < end_s:
            time.sleep(COMMAND_PERIOD_S)
            assert not controller.tick(), "watchdog tripped: command loop too slow"
            controller.command(CIRCLE)
        controller.stop()
        assert not controller.is_moving

    assert _ask("Did the robot drive a counter-clockwise (left-turning) arc and then stop?")
