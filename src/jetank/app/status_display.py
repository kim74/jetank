"""
Runs the status display alongside other work, without letting it block or stop the robot.

Layer: app
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from jetank.errors import HardwareError

if TYPE_CHECKING:
    from collections.abc import Callable

    from jetank.control.status import StatusReporter
    from jetank.domain.status import RobotMode
    from jetank.interfaces.display import Display

logger = logging.getLogger(__name__)


async def run_status_display(
    reporter: StatusReporter, mode: Callable[[], RobotMode], period_s: float
) -> None:
    """Refresh the display every ``period_s`` until cancelled.

    The I2C transfer runs in a worker thread so it never delays the event loop (and with it
    the drive watchdog). The display is optional: if it fails, the failure is logged and the
    loop ends, but nothing else is affected.

    Args:
        reporter: Gathers and shows the status.
        mode: Called on the event loop each refresh to get the current mode.
        period_s: Time between refreshes.
    """
    while True:
        try:
            await asyncio.to_thread(reporter.update, mode())
        except HardwareError as exc:
            logger.warning("status display failed, no longer updating it: %s", exc)
            return
        await asyncio.sleep(period_s)


def show_status_until_interrupted(
    reporter: StatusReporter,
    mode: RobotMode,
    period_s: float,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """Refresh the display every ``period_s`` until Ctrl-C (or SIGHUP/SIGTERM, if mapped).

    Args:
        sleep: Delay function; defaults to :func:`time.sleep`, looked up at call time.

    Raises:
        HardwareError: The display could not be written.
    """
    pause = sleep if sleep is not None else time.sleep
    try:
        while True:
            reporter.update(mode)
            pause(period_s)
    except KeyboardInterrupt as exc:
        logger.info("status display stopped: %s", exc or "Ctrl-C")


def close_display_quietly(display: Display) -> None:
    """Close an optional display, logging instead of raising if it fails."""
    try:
        display.close()
    except HardwareError as exc:
        logger.warning("could not blank the status display: %s", exc)
