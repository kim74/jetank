"""
Status display: gathers hostname, IP address, battery and mode, and shows them.

Layer: control
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.domain.status import RobotStatus, format_status
from jetank.errors import HardwareError

if TYPE_CHECKING:
    from jetank.domain.status import BatteryState, RobotMode, StatusScreen
    from jetank.interfaces.display import Display
    from jetank.interfaces.host import HostInfo
    from jetank.interfaces.power import PowerMonitor

logger = logging.getLogger(__name__)


class StatusReporter:
    """Keeps a :class:`Display` showing the current :class:`RobotStatus`.

    The display, power monitor and host info are borrowed, not owned; the caller closes them.
    A failing battery monitor is shown as an unknown voltage rather than raised, because the
    status display must keep working without it.

    Args:
        display: Where the status is shown.
        power: Battery monitor.
        host: Source of hostname and IP address.
        low_voltage_v: At or below this voltage the battery is flagged as low.
    """

    def __init__(
        self, display: Display, power: PowerMonitor, host: HostInfo, low_voltage_v: float
    ) -> None:
        self._display = display
        self._power = power
        self._host = host
        self._low_voltage_v = low_voltage_v
        self._shown: StatusScreen | None = None
        self._battery_failed = False

    def update(self, mode: RobotMode) -> StatusScreen:
        """Read the current status and show it if it differs from what is on display.

        Returns:
            The screen now on display.

        Raises:
            HardwareError: The display could not be written.
        """
        status = RobotStatus(
            hostname=self._host.hostname(),
            ip_address=self._host.ip_address(),
            battery=self._read_battery(),
            mode=mode,
        )
        screen = format_status(
            status, self._display.columns, self._display.rows, self._low_voltage_v
        )
        if screen != self._shown:
            # Forget the old screen first: if the write fails, the next update retries it.
            self._shown = None
            self._display.show(screen)
            self._shown = screen
        return screen

    def _read_battery(self) -> BatteryState | None:
        try:
            battery = self._power.read_battery()
        except HardwareError as exc:
            if not self._battery_failed:
                logger.warning("battery monitor unavailable: %s", exc)
            self._battery_failed = True
            return None
        if self._battery_failed:
            logger.info("battery monitor available again")
        self._battery_failed = False
        return battery
