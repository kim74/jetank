from __future__ import annotations

import logging

import pytest

from jetank.control.status import StatusReporter
from jetank.domain.status import RobotMode
from jetank.drivers.fake import FakeDisplay, FakeHostInfo, FakePowerMonitor
from jetank.errors import HardwareError


@pytest.fixture
def display() -> FakeDisplay:
    return FakeDisplay(columns=21, rows=4)


@pytest.fixture
def power() -> FakePowerMonitor:
    return FakePowerMonitor(voltage_v=11.9)


@pytest.fixture
def host() -> FakeHostInfo:
    return FakeHostInfo(hostname="jetank", ip_address="192.168.1.42")


@pytest.fixture
def reporter(display: FakeDisplay, power: FakePowerMonitor, host: FakeHostInfo) -> StatusReporter:
    return StatusReporter(display, power, host, low_voltage_v=10.5)


def test_update_shows_current_status(reporter: StatusReporter, display: FakeDisplay) -> None:
    reporter.update(RobotMode.TELEOP_WAITING)
    assert display.current.lines == ("jetank", "192.168.1.42", "bat 11.9V", "teleop: waiting")


def test_unchanged_status_is_not_rewritten(reporter: StatusReporter, display: FakeDisplay) -> None:
    reporter.update(RobotMode.IDLE)
    reporter.update(RobotMode.IDLE)
    assert len(display.history) == 1


def test_changes_are_shown(
    reporter: StatusReporter, display: FakeDisplay, power: FakePowerMonitor, host: FakeHostInfo
) -> None:
    reporter.update(RobotMode.IDLE)
    power.voltage_v = 10.4
    host.address = None
    reporter.update(RobotMode.TELEOP_DRIVING)
    assert display.current.lines == ("jetank", "no network", "bat 10.4V LOW", "teleop: driving")
    assert len(display.history) == 2


def test_screen_fits_the_display(power: FakePowerMonitor, host: FakeHostInfo) -> None:
    small = FakeDisplay(columns=8, rows=2)
    StatusReporter(small, power, host, low_voltage_v=10.5).update(RobotMode.IDLE)
    assert small.current.lines == ("jetank", "192.168.")


def test_battery_failure_shows_unknown_and_logs_once(
    reporter: StatusReporter,
    display: FakeDisplay,
    power: FakePowerMonitor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    power.fail_with = HardwareError("INA219 gone")
    with caplog.at_level(logging.WARNING):
        reporter.update(RobotMode.IDLE)
        reporter.update(RobotMode.IDLE)
    assert display.current.lines[2] == "bat --.-V"
    assert [r.message for r in caplog.records].count(
        "battery monitor unavailable: INA219 gone"
    ) == 1


def test_battery_recovery_is_shown(
    reporter: StatusReporter, display: FakeDisplay, power: FakePowerMonitor
) -> None:
    power.fail_with = HardwareError("glitch")
    reporter.update(RobotMode.IDLE)
    power.fail_with = None
    reporter.update(RobotMode.IDLE)
    assert display.current.lines[2] == "bat 11.9V"


def test_display_failure_propagates_and_is_retried(
    reporter: StatusReporter, display: FakeDisplay
) -> None:
    display.fail_with = HardwareError("OLED gone")
    with pytest.raises(HardwareError, match="OLED gone"):
        reporter.update(RobotMode.IDLE)
    display.fail_with = None
    reporter.update(RobotMode.IDLE)
    assert display.current.lines[0] == "jetank"
