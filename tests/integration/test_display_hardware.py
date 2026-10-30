"""Bring-up test for the OLED (SSD1306) and the battery monitor (INA219). Nothing moves.

Run on the robot, watching the OLED:

    pytest -m hardware tests/integration/test_display_hardware.py -s

The chips answering on I2C is checked automatically. Whether the text is readable and the
right way up can only be seen: with ``-s`` the operator is asked; without it those
questions are skipped.

Bus and addresses come from ``config/jetank.yaml`` if it exists, otherwise the defaults; the
backends are forced to ``hardware``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest

from jetank.config.settings import DisplayConfig, PowerConfig, load_settings
from jetank.control.status import StatusReporter
from jetank.domain.status import RobotMode, StatusScreen
from jetank.drivers.ina219 import Ina219PowerMonitor
from jetank.drivers.ssd1306 import Ssd1306Display
from jetank.drivers.system_host import SystemHostInfo
from jetank.errors import DriverClosedError

pytestmark = pytest.mark.hardware

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config" / "jetank.yaml"
# A 3S pack reads between ~9 V (empty) and 12.6 V (full); USB-only power reads near 0.
PLAUSIBLE_BATTERY_V: Final = (8.0, 13.0)

Confirm = Callable[[str], None]


def _settings() -> tuple[DisplayConfig, PowerConfig]:
    settings = load_settings(CONFIG_PATH if CONFIG_PATH.exists() else None)
    return (
        settings.display.model_copy(update={"backend": "hardware"}),
        settings.power.model_copy(update={"backend": "hardware"}),
    )


@pytest.fixture
def confirm(pytestconfig: pytest.Config) -> Confirm:
    """Ask the operator a yes/no question; a "no" fails the test. No-op without ``-s``."""
    interactive = pytestconfig.getoption("capture") == "no"

    def _confirm(question: str) -> None:
        if not interactive:
            return
        answer = input(f"\n{question} [y/n] ").strip().lower()
        assert answer == "y", f"operator answered no to: {question}"

    return _confirm


@pytest.fixture
def display() -> Iterator[Ssd1306Display]:
    """Real panel; closing it on teardown blanks and switches it off even if a test failed."""
    with Ssd1306Display.open(_settings()[0]) as panel:
        yield panel


@pytest.fixture
def power() -> Iterator[Ina219PowerMonitor]:
    with Ina219PowerMonitor.open(_settings()[1]) as monitor:
        yield monitor


def test_battery_voltage_is_plausible(power: Ina219PowerMonitor) -> None:
    voltage_v = power.read_battery().voltage_v
    low, high = PLAUSIBLE_BATTERY_V
    assert low <= voltage_v <= high, f"battery reads {voltage_v:.2f} V; is the pack switched on?"


def test_display_shows_text_the_right_way_up(display: Ssd1306Display, confirm: Confirm) -> None:
    display.show(StatusScreen(lines=("TOP LINE", "line 2", "line 3", "BOTTOM LINE")))
    confirm("Does the OLED show 'TOP LINE' at the top, readable and not mirrored?")


def test_status_screen(
    display: Ssd1306Display, power: Ina219PowerMonitor, confirm: Confirm
) -> None:
    low_voltage_v = _settings()[1].low_voltage_v
    screen = StatusReporter(display, power, SystemHostInfo(), low_voltage_v).update(RobotMode.IDLE)
    confirm(f"Does the OLED show {' / '.join(screen.lines)}?")


def test_close_blanks_the_display(display: Ssd1306Display, confirm: Confirm) -> None:
    display.show(StatusScreen(lines=("about to blank",)))
    display.close()
    with pytest.raises(DriverClosedError):
        display.clear()
    confirm("Is the OLED dark now?")
