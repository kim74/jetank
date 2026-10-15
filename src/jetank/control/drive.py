"""
Drive controller: turns drive commands into track speeds, enforces the speed limit and
stops the robot when commands stop arriving.

Layer: control
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from jetank.domain.kinematics import mix

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.settings import DriveConfig
    from jetank.domain.models import DriveCommand, TrackSpeeds
    from jetank.interfaces.clock import Clock
    from jetank.interfaces.motors import MotorDriver

logger = logging.getLogger(__name__)


class DriveController:
    """Safe front end to a :class:`~jetank.interfaces.MotorDriver`.

    Every command is scaled by ``config.max_speed``. The caller must invoke :meth:`tick`
    periodically (well within ``config.watchdog_timeout_s``); if the last non-zero command
    is older than the timeout, the motors are stopped. The controller does not own the
    motor driver and never closes it.

    All public methods are thread-safe, so commands and ticks may come from different threads.
    """

    def __init__(self, motors: MotorDriver, clock: Clock, config: DriveConfig) -> None:
        self._motors = motors
        self._clock = clock
        self._config = config
        self._lock = threading.Lock()
        self._last_command_s = clock.monotonic()
        self._moving = False
        self._motors.stop()

    @property
    def is_moving(self) -> bool:
        """True while a non-zero command is applied and the watchdog has not tripped."""
        return self._moving

    def __enter__(self) -> DriveController:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def command(self, command: DriveCommand) -> TrackSpeeds:
        """Apply a body-frame command. Returns the track speeds actually sent."""
        return self.set_tracks(mix(command))

    def set_tracks(self, speeds: TrackSpeeds) -> TrackSpeeds:
        """Apply track speeds directly. Returns the speeds actually sent (after limiting)."""
        limited = speeds.scaled(self._config.max_speed)
        with self._lock:
            try:
                self._motors.set_speeds(limited)
            except Exception:
                self._emergency_stop()
                raise
            self._last_command_s = self._clock.monotonic()
            self._moving = not limited.is_stopped
        return limited

    def stop(self) -> None:
        """Stop both tracks."""
        with self._lock:
            self._motors.stop()
            self._moving = False

    def tick(self) -> bool:
        """Run the watchdog check.

        Returns:
            True if this call stopped the motors because the last command timed out.
        """
        with self._lock:
            if not self._moving:
                return False
            age_s = self._clock.monotonic() - self._last_command_s
            if age_s <= self._config.watchdog_timeout_s:
                return False
            logger.warning(
                "drive watchdog: no command for %.3f s (limit %.3f s), stopping",
                age_s,
                self._config.watchdog_timeout_s,
            )
            self._motors.stop()
            self._moving = False
            return True

    def _emergency_stop(self) -> None:
        self._moving = False
        try:
            self._motors.stop()
        except Exception:
            logger.exception("failed to stop motors after a failed command")
