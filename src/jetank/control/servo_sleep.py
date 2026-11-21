"""
Sleep timers for the arm: when it counts as idle, and when a homed arm may go limp.

Layer: control

Non-blocking: :meth:`ServoSleep.fall_asleep` starts the arm's way home and
:meth:`ServoSleep.tick`, called periodically, releases it once the move is done. The owner
(e.g. :class:`~jetank.control.teleop.TeleopSession`) decides what else sleep means, such as
stopping the tracks or letting the gripper go limp.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.control.arm import PARK_SETTLE_S
from jetank.errors import HardwareError

if TYPE_CHECKING:
    from jetank.control.arm import ArmController
    from jetank.interfaces.clock import Clock

logger = logging.getLogger(__name__)


class ServoSleep:
    """Puts the arm to sleep on request or after a time without servo commands.

    Not thread-safe: call it from the thread that owns the arm.

    Args:
        arm: Arm, or ``None`` if the robot has none (then only the idle timer runs).
        clock: Time source.
        idle_sleep_s: Seconds without a servo command after which :meth:`tick` asks for
            sleep while a servo is powered; 0 disables the automatic sleep.

    Raises:
        ValueError: ``idle_sleep_s`` is negative.
    """

    def __init__(self, arm: ArmController | None, clock: Clock, idle_sleep_s: float) -> None:
        if idle_sleep_s < 0:
            raise ValueError("idle sleep time must not be negative")
        self._arm = arm
        self._clock = clock
        self._idle_sleep_s = idle_sleep_s
        self._last_command_s = clock.monotonic()
        self._release_at_s: float | None = None
        self._asleep = False

    @property
    def asleep(self) -> bool:
        """True from :meth:`fall_asleep` until the next :meth:`wake`."""
        return self._asleep

    @property
    def idle_sleep_s(self) -> float:
        """Seconds without a servo command before the automatic sleep; 0 = never."""
        return self._idle_sleep_s

    def wake(self) -> None:
        """A servo command arrived: stay awake and cancel a release still pending."""
        self._asleep = False
        self._release_at_s = None
        self._last_command_s = self._clock.monotonic()

    def fall_asleep(self) -> str | None:
        """Start moving a powered arm home; :meth:`tick` lets it go limp once it is there.

        Restarts the idle timer, so a failure is retried only after another ``idle_sleep_s``.

        Returns:
            An error text if the arm could not move home (it then keeps holding, since going
            limp away from home could let it fall), else ``None``.
        """
        self._asleep = True
        self._last_command_s = self._clock.monotonic()
        if self._arm is None or not self._arm.engaged:
            return None
        try:
            duration_s = self._arm.rest()
        except HardwareError as exc:
            logger.warning("sleep: arm could not move home, it keeps holding: %s", exc)
            return f"arm failed: {exc}"
        self._release_at_s = self._clock.monotonic() + duration_s + PARK_SETTLE_S
        return None

    def tick(self, servos_powered: bool) -> bool:
        """Release a homed arm when due, and check the idle timer.

        Args:
            servos_powered: Whether any servo (arm or gripper) holds torque now.

        Returns:
            True if the robot has been idle for ``idle_sleep_s`` with a servo powered, so the
            caller should put it to sleep. Never True while a release is pending.
        """
        now_s = self._clock.monotonic()
        if self._release_at_s is not None:
            if now_s >= self._release_at_s:
                self._release_at_s = None
                self._release_arm()
            return False
        return (
            self._idle_sleep_s > 0
            and servos_powered
            and now_s - self._last_command_s >= self._idle_sleep_s
        )

    def _release_arm(self) -> None:
        assert self._arm is not None  # a release is only scheduled for an engaged arm
        try:
            self._arm.release()
        except HardwareError as exc:
            logger.error("sleep: arm could not go limp: %s", exc)
