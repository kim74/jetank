"""
Gripper: opens and closes one bus servo within its calibrated limits.

Layer: control

Safety rules:

* Every target is clamped to the calibrated limits, except closing (opening 0.0), which aims
  ``grip_overshoot`` counts past the closed limit to take up play in the jaw mechanics.
* Before torque is switched on, the servo's goal is set to where it is now, so switching
  torque on cannot make it jump (it may rest outside the limits while limp).
* Moves take time in proportion to their length (``move_time_s`` for a full stroke).
* The servo's torque is capped (``max_torque``) before it is switched on, so closing on an
  object squeezes it with bounded force and a stalled servo draws bounded current.
* Only the gripper's own servo is ever touched.
* :meth:`GripperController.close` switches the torque off: the gripper goes limp.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from jetank.domain.servo import GripperStatus
from jetank.errors import DriverClosedError

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.domain.servo import GripperCalibration
    from jetank.interfaces.servo import ServoBus

logger = logging.getLogger(__name__)

#: Shortest move time sent to the servo, so even tiny corrections are not snapped.
MIN_MOVE_S: Final = 0.1
#: Duration used to park the goal at the present position before torque is switched on.
_PARK_S: Final = 0.1


class GripperController:
    """Drives the gripper servo. The bus is borrowed, not owned.

    Args:
        bus: Servo bus the gripper is on.
        calibration: Servo ID and open/closed limits.
        move_time_s: Time for a full open-to-closed stroke.
        max_torque: Torque cap as a fraction (0, 1] of the servo's maximum.
    """

    def __init__(
        self,
        bus: ServoBus,
        calibration: GripperCalibration,
        move_time_s: float,
        max_torque: float = 1.0,
    ) -> None:
        if move_time_s <= 0:
            raise ValueError("move time must be positive")
        if not 0.0 < max_torque <= 1.0:
            raise ValueError(f"max torque must be within (0, 1], got {max_torque}")
        self._max_torque = max_torque
        self._bus = bus
        self._calibration = calibration
        self._move_time_s = move_time_s
        self._engaged = False
        self._closed = False

    def __enter__(self) -> GripperController:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def servo_id(self) -> int:
        return self._calibration.servo_id

    def status(self) -> GripperStatus:
        """Read the gripper without moving it.

        Raises:
            ServoError: The servo did not answer.
        """
        self._ensure_open()
        servo = self._bus.read_state(self.servo_id)
        return GripperStatus(opening=self._calibration.opening(servo.position), servo=servo)

    def open(self) -> int:
        """Move to the open limit. Returns the target position."""
        return self.set_opening(1.0)

    def shut(self) -> int:
        """Close the jaws (aiming at the grip position). Returns the target position."""
        return self.set_opening(0.0)

    def set_opening(self, opening: float) -> int:
        """Move to ``opening`` (0.0 = closed, 1.0 = open). Does not wait for the move.

        Closing (0.0) aims at the calibration's grip position, past the closed limit, so the
        jaws press on whatever they hold with the force the torque cap allows.

        Returns:
            The target position sent to the servo.

        Raises:
            ValueError: ``opening`` is outside [0, 1].
            ServoError: The servo did not answer or reported a fault.
        """
        self._ensure_open()
        cal = self._calibration
        target = cal.grip_position if opening == 0.0 else cal.clamp(cal.target(opening))
        present = self._bus.read_state(self.servo_id).position
        if not self._engaged:
            self._engage(present)
        self._bus.move(self.servo_id, target, self._duration(present, target))
        logger.info("gripper: %d -> %d (opening %.2f)", present, target, opening)
        return target

    def release(self) -> None:
        """Switch the torque off; the gripper goes limp and lets go of anything held.

        Raises:
            ServoError: The servo did not acknowledge.
        """
        self._ensure_open()
        self._bus.set_torque(self.servo_id, False)
        self._engaged = False

    def close(self) -> None:
        """Release the gripper if this controller engaged it. Idempotent.

        A controller that only read the status sends nothing. The bus is left open for
        its owner.
        """
        if self._closed:
            return
        try:
            if self._engaged:
                self.release()
        finally:
            self._closed = True

    def _engage(self, present: int) -> None:
        # Counted as engaged before any reply: a lost acknowledgement may hide a servo that
        # did switch its torque on (some firmware does so on a goal write), and close()
        # must still switch it off.
        self._engaged = True
        self._bus.set_torque_limit(self.servo_id, self._max_torque)
        self._bus.move(self.servo_id, present, _PARK_S)
        self._bus.set_torque(self.servo_id, True)

    def _duration(self, start: int, target: int) -> float:
        cal = self._calibration
        stroke = abs(cal.open_position - cal.closed_position)
        return max(MIN_MOVE_S, self._move_time_s * abs(target - start) / stroke)

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("gripper controller is closed")
