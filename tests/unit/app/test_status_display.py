from __future__ import annotations

import asyncio

import pytest

from jetank.app.status_display import (
    close_display_quietly,
    run_status_display,
    show_status_until_interrupted,
)
from jetank.control.status import StatusReporter
from jetank.domain.status import RobotMode
from jetank.drivers.fake import FakeDisplay, FakeHostInfo, FakePowerMonitor
from jetank.errors import HardwareError


@pytest.fixture
def display() -> FakeDisplay:
    return FakeDisplay()


@pytest.fixture
def reporter(display: FakeDisplay) -> StatusReporter:
    return StatusReporter(display, FakePowerMonitor(), FakeHostInfo(), low_voltage_v=10.5)


def test_async_loop_follows_mode_until_cancelled(
    reporter: StatusReporter, display: FakeDisplay
) -> None:
    modes = iter([RobotMode.TELEOP_WAITING, RobotMode.TELEOP_DRIVING])

    async def scenario() -> None:
        task = asyncio.create_task(run_status_display(reporter, lambda: next(modes), 0.001))
        while len(display.history) < 2:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert [s.lines[-1] for s in display.history] == ["teleop: waiting", "teleop: driving"]


def test_async_loop_ends_quietly_when_display_fails(
    reporter: StatusReporter, display: FakeDisplay, caplog: pytest.LogCaptureFixture
) -> None:
    display.fail_with = HardwareError("OLED gone")
    asyncio.run(run_status_display(reporter, lambda: RobotMode.IDLE, 0.001))
    assert "no longer updating" in caplog.text


def test_blocking_loop_refreshes_until_interrupted(
    reporter: StatusReporter, display: FakeDisplay
) -> None:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise KeyboardInterrupt

    show_status_until_interrupted(reporter, RobotMode.IDLE, 2.0, sleep=sleep)
    assert sleeps == [2.0, 2.0, 2.0]
    assert display.current.lines[-1] == "idle"


def test_close_quietly_logs_display_failure(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenDisplay(FakeDisplay):
        def close(self) -> None:
            raise HardwareError("bus error")

    close_display_quietly(BrokenDisplay())
    assert "could not blank" in caplog.text
