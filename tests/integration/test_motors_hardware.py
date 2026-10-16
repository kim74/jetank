"""Bring-up test for the real track motors (PCA9685 -> TB6612 on the JETANK).

Lift the tracks off the ground, then run on the robot:

    JETANK_ALLOW_MOTION=1 pytest -m hardware tests/integration/test_motors_hardware.py -s

Every test reads the PCA9685 output registers back over I2C, so wiring to the chip is
verified automatically. Whether the right track turns the right way can only be seen: with
``-s`` the operator is asked after each pulse; without it those questions are skipped.

Settings come from ``config/jetank.yaml`` if it exists, otherwise the defaults, and
``JETANK_*`` environment variables override both (e.g. ``JETANK_MOTORS__I2C_BUS=1``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest
from smbus2 import SMBus

from jetank.config.settings import DriveConfig, MotorsConfig, load_settings
from jetank.control.drive import DriveController
from jetank.domain.models import DriveCommand, TrackSpeeds
from jetank.drivers.pca9685 import LED0_ON_L, duty_to_block
from jetank.drivers.system_clock import SystemClock
from jetank.drivers.tb6612_tracks import Tb6612TrackDriver, bridge_outputs
from jetank.errors import DriverClosedError

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("JETANK_ALLOW_MOTION") != "1",
        reason="set JETANK_ALLOW_MOTION=1 to allow the motors to move",
    ),
]

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "jetank.yaml"
PULSE_SPEED: Final = 0.3
PULSE_S: Final = 0.5
WATCHDOG_TIMEOUT_S: Final = 0.3
# Allowed lateness of the watchdog stop: tick period plus scheduling jitter on the Jetson.
WATCHDOG_SLACK_S: Final = 0.1
TICK_PERIOD_S: Final = 0.01

Confirm = Callable[[str], None]


@pytest.fixture(scope="module")
def config() -> MotorsConfig:
    return load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None).motors


@pytest.fixture(scope="module")
def interactive(pytestconfig: pytest.Config) -> bool:
    """True when output capture is off (``-s``), so the operator can answer prompts."""
    interactive = bool(pytestconfig.getoption("capture") == "no")
    if interactive:
        answer = input("\nAre the tracks lifted off the ground? [y/n] ").strip().lower()
        if answer != "y":
            pytest.skip("tracks not lifted; motor tests not run")
    return interactive


@pytest.fixture
def confirm(interactive: bool) -> Confirm:
    """Ask the operator a yes/no question; a "no" fails the test. No-op without ``-s``."""

    def _confirm(question: str) -> None:
        if not interactive:
            return
        answer = input(f"\n{question} [y/n] ").strip().lower()
        assert answer == "y", f"operator answered no to: {question}"

    return _confirm


@pytest.fixture
def driver(config: MotorsConfig, interactive: bool) -> Iterator[Tb6612TrackDriver]:
    """Real track driver; closing it on teardown stops the motors even if the test failed."""
    with Tb6612TrackDriver.open(config) as track_driver:
        yield track_driver


def _motor_channels(config: MotorsConfig) -> list[int]:
    return sorted(config.left.channels | config.right.channels)


def _read_blocks(config: MotorsConfig) -> dict[int, list[int]]:
    """Read the 4 LEDn_ON_L..LEDn_OFF_H bytes of every motor channel from the chip."""
    with SMBus(config.i2c_bus) as bus:
        return {
            channel: bus.read_i2c_block_data(config.i2c_address, LED0_ON_L + 4 * channel, 4)
            for channel in _motor_channels(config)
        }


def _expected_blocks(config: MotorsConfig, speeds: TrackSpeeds) -> dict[int, list[int]]:
    expected: dict[int, list[int]] = {}
    for channel, speed in ((config.left, speeds.left), (config.right, speeds.right)):
        outputs = bridge_outputs(speed, channel)
        expected[channel.pwm_channel] = duty_to_block(outputs.pwm_duty)
        expected[channel.forward_channel] = duty_to_block(1.0 if outputs.forward else 0.0)
        expected[channel.reverse_channel] = duty_to_block(1.0 if outputs.reverse else 0.0)
    return expected


def _assert_outputs_off(config: MotorsConfig) -> None:
    assert _read_blocks(config) == _expected_blocks(config, TrackSpeeds.stopped())


def test_driver_opens_with_outputs_off(config: MotorsConfig, driver: Tb6612TrackDriver) -> None:
    _assert_outputs_off(config)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("track", ["left", "right"])
def test_single_track_pulse(
    config: MotorsConfig,
    driver: Tb6612TrackDriver,
    confirm: Confirm,
    track: str,
    direction: str,
) -> None:
    speed = PULSE_SPEED if direction == "forward" else -PULSE_SPEED
    speeds = TrackSpeeds(
        left=speed if track == "left" else 0.0,
        right=speed if track == "right" else 0.0,
    )

    driver.set_speeds(speeds)
    try:
        assert _read_blocks(config) == _expected_blocks(config, speeds)
        time.sleep(PULSE_S)
    finally:
        driver.stop()
    _assert_outputs_off(config)

    confirm(
        f"Did ONLY the {track.upper()} track run {direction.upper()}? "
        f"(wrong direction -> set motors.{track}.inverted; wrong track -> swap channels)"
    )


def test_drive_controller_turns_left_then_watchdog_stops(
    config: MotorsConfig, driver: Tb6612TrackDriver, confirm: Confirm
) -> None:
    clock = SystemClock()
    drive_config = DriveConfig(max_speed=PULSE_SPEED, watchdog_timeout_s=WATCHDOG_TIMEOUT_S)
    controller = DriveController(driver, clock, drive_config)

    # Positive angular is a counter-clockwise (left) turn: left track back, right forward.
    sent = controller.command(DriveCommand(angular=1.0))
    assert sent.left < 0.0 < sent.right
    assert _read_blocks(config) == _expected_blocks(config, sent)

    # No further commands: the watchdog alone must stop the tracks.
    commanded_s = clock.monotonic()
    deadline_s = commanded_s + WATCHDOG_TIMEOUT_S + 1.0
    while not controller.tick():
        assert clock.monotonic() < deadline_s, "watchdog never stopped the motors"
        time.sleep(TICK_PERIOD_S)
    elapsed_s = clock.monotonic() - commanded_s

    assert not controller.is_moving
    assert WATCHDOG_TIMEOUT_S <= elapsed_s <= WATCHDOG_TIMEOUT_S + WATCHDOG_SLACK_S
    _assert_outputs_off(config)

    confirm(
        "Did the robot turn LEFT (left track back, right track forward) "
        f"and stop by itself after about {WATCHDOG_TIMEOUT_S} s?"
    )


def test_exception_while_driving_leaves_motors_stopped(
    config: MotorsConfig, interactive: bool
) -> None:
    opened: list[Tb6612TrackDriver] = []

    def drive_then_fail() -> None:
        with Tb6612TrackDriver.open(config) as track_driver:
            opened.append(track_driver)
            track_driver.set_speeds(TrackSpeeds(left=PULSE_SPEED, right=PULSE_SPEED))
            raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        drive_then_fail()

    _assert_outputs_off(config)
    with pytest.raises(DriverClosedError):
        opened[0].set_speeds(TrackSpeeds.stopped())
