from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jetank.domain.status import (
    BatteryState,
    RobotMode,
    RobotStatus,
    StatusScreen,
    format_status,
)


def make_status(
    hostname: str = "jetank",
    ip_address: str | None = "192.168.1.42",
    voltage_v: float | None = 11.93,
    mode: RobotMode = RobotMode.IDLE,
) -> RobotStatus:
    battery = None if voltage_v is None else BatteryState(voltage_v=voltage_v)
    return RobotStatus(hostname=hostname, ip_address=ip_address, battery=battery, mode=mode)


def test_four_rows_show_host_ip_battery_and_mode() -> None:
    screen = format_status(make_status(mode=RobotMode.TELEOP_DRIVING), 21, 4, 10.5)
    assert screen.lines == ("jetank", "192.168.1.42", "bat 11.9V", "teleop: driving")


def test_missing_network_and_battery_are_shown_as_unknown() -> None:
    screen = format_status(make_status(ip_address=None, voltage_v=None), 21, 4, 10.5)
    assert screen.lines[1:3] == ("no network", "bat --.-V")


@pytest.mark.parametrize(("voltage_v", "flagged"), [(10.5, True), (9.0, True), (10.6, False)])
def test_low_battery_is_flagged_at_or_below_threshold(voltage_v: float, flagged: bool) -> None:
    line = format_status(make_status(voltage_v=voltage_v), 21, 4, 10.5).lines[2]
    assert line.endswith(" LOW") is flagged


def test_fewer_rows_drop_the_last_lines() -> None:
    assert format_status(make_status(), 21, 2, 10.5).lines == ("jetank", "192.168.1.42")


def test_long_lines_are_cut_to_the_column_count() -> None:
    screen = format_status(make_status(hostname="a-very-long-hostname-indeed"), 21, 4, 10.5)
    assert screen.lines[0] == "a-very-long-hostname-"


def test_unprintable_hostname_characters_are_replaced() -> None:
    assert format_status(make_status(hostname="jé\ttank"), 21, 4, 10.5).lines[0] == "j??tank"


@pytest.mark.parametrize(("columns", "rows"), [(0, 4), (21, 0)])
def test_non_positive_grid_is_rejected(columns: int, rows: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        format_status(make_status(), columns, rows, 10.5)


@given(
    hostname=st.text(max_size=40),
    columns=st.integers(min_value=1, max_value=40),
    rows=st.integers(min_value=1, max_value=8),
    voltage_v=st.none() | st.floats(min_value=0.0, max_value=32.0),
)
def test_every_status_fits_the_grid(
    hostname: str, columns: int, rows: int, voltage_v: float | None
) -> None:
    screen = format_status(make_status(hostname=hostname, voltage_v=voltage_v), columns, rows, 10.5)
    assert len(screen.lines) <= rows
    assert all(len(line) <= columns for line in screen.lines)


def test_screen_rejects_unprintable_text() -> None:
    with pytest.raises(ValidationError, match="cannot show"):
        StatusScreen(lines=("tab\there",))


def test_screen_rejects_too_many_lines() -> None:
    with pytest.raises(ValidationError):
        StatusScreen(lines=("x",) * 9)


@pytest.mark.parametrize("voltage_v", [-0.1, 32.1, float("nan")])
def test_battery_voltage_outside_sensor_range_is_rejected(voltage_v: float) -> None:
    with pytest.raises(ValidationError):
        BatteryState(voltage_v=voltage_v)
