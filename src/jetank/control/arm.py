"""
Arm: moves the arm joints' bus servos within their calibrated limits.

Layer: control

Safety rules:

* Every target is clamped to the joint's calibrated limits; joints without a calibration
  can never be commanded.
* Before a joint's torque is switched on, its goal is set to where it is now, so switching
  torque on cannot make it jump (a limp joint may rest outside its limits).
* Moves run at a bounded speed (``speed_cps``), however far the target.
* Only the configured joints' servos are ever touched.
* :meth:`ArmController.close` switches the torque of every engaged joint off: the arm goes
  limp and sags under gravity, so callers move it home first when they can.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from jetank.errors import ConfigurationError, DriverClosedError, ServoError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from jetank.domain.servo import JointCalibration, JointName, ServoState
    from jetank.interfaces.servo import ServoBus

logger = logging.getLogger(__name__)

#: Shortest move time sent to a servo, so even tiny corrections are not snapped.
MIN_MOVE_S: Final = 0.1
#: Longest move time the SCS goal-time register can hold.
MAX_MOVE_S: Final = 65.0
#: Duration used to park the goal at the present position before torque is switched on.
_PARK_S: Final = 0.1


class ArmController:
    """Drives the arm joints. The bus is borrowed, not owned.

    Args:
        bus: Servo bus the joints are on.
        joints: Calibration per joint; joints left out cannot be moved.
        speed_cps: Joint speed in servo counts per second.

    Raises:
        ValueError: ``speed_cps`` is not positive.
    """

    def __init__(
        self,
        bus: ServoBus,
        joints: Mapping[JointName, JointCalibration],
        speed_cps: float,
    ) -> None:
        if speed_cps <= 0:
            raise ValueError("speed must be positive")
        self._bus = bus
        self._joints = dict(joints)
        self._speed_cps = speed_cps
        self._engaged: set[JointName] = set()
        self._closed = False

    def __enter__(self) -> ArmController:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def joints(self) -> Mapping[JointName, JointCalibration]:
        """Calibration of every joint this controller may move."""
        return self._joints

    @property
    def engaged(self) -> bool:
        """Whether this controller has switched torque on for any joint."""
        return bool(self._engaged)

    def state(self, joint: JointName) -> ServoState:
        """Read one joint's servo without moving it.

        Raises:
            ConfigurationError: ``joint`` has no calibration.
            ServoError: The servo did not answer.
        """
        self._ensure_open()
        return self._bus.read_state(self._calibration(joint).servo_id)

    def hold(self) -> None:
        """Switch torque on for every joint where it stands now, without moving it.

        Raises:
            ServoError: A servo did not answer or reported a fault.
        """
        self._ensure_open()
        for joint in self._joints:
            if joint not in self._engaged:
                self._engage(joint, self.state(joint).position)

    def move(self, joint: JointName, position: int) -> float:
        """Start moving ``joint`` towards ``position``, clamped to its limits. Does not wait.

        Returns:
            The move's duration in seconds.

        Raises:
            ConfigurationError: ``joint`` has no calibration.
            ServoError: The servo did not answer or reported a fault.
        """
        self._ensure_open()
        cal = self._calibration(joint)
        target = cal.clamp(position)
        if target != position:
            logger.warning("arm: %s target %d clamped to %d", joint.value, position, target)
        present = self._bus.read_state(cal.servo_id).position
        if joint not in self._engaged:
            self._engage(joint, present)
        duration_s = min(MAX_MOVE_S, max(MIN_MOVE_S, abs(target - present) / self._speed_cps))
        self._bus.move(cal.servo_id, target, duration_s)
        logger.info("arm: %s %d -> %d in %.2f s", joint.value, present, target, duration_s)
        return duration_s

    def home(self) -> float:
        """Start moving every joint to its home position. Does not wait.

        Returns:
            Duration of the longest of the moves, in seconds.
        """
        return max(
            (self.move(joint, cal.home_position) for joint, cal in self._joints.items()),
            default=0.0,
        )

    def release(self) -> None:
        """Switch the torque of every engaged joint off; the arm goes limp.

        Every joint is tried even if one fails; the first failure is raised afterwards.

        Raises:
            ServoError: A servo did not acknowledge.
        """
        self._ensure_open()
        first_error: ServoError | None = None
        for joint in sorted(self._engaged, key=lambda j: j.value):
            try:
                self._bus.set_torque(self._joints[joint].servo_id, False)
                self._engaged.discard(joint)
            except ServoError as exc:  # keep releasing the others first
                logger.error("arm: could not release %s: %s", joint.value, exc)
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        """Release every joint this controller engaged. Idempotent.

        A controller that only read states sends nothing. The bus is left open for its owner.
        """
        if self._closed:
            return
        try:
            if self._engaged:
                self.release()
        finally:
            self._closed = True

    def _engage(self, joint: JointName, present: int) -> None:
        # Counted as engaged before any reply: a lost acknowledgement may hide a servo that
        # did switch its torque on, and close() must still switch it off.
        self._engaged.add(joint)
        servo_id = self._joints[joint].servo_id
        self._bus.move(servo_id, present, _PARK_S)
        self._bus.set_torque(servo_id, True)

    def _calibration(self, joint: JointName) -> JointCalibration:
        try:
            return self._joints[joint]
        except KeyError:
            raise ConfigurationError(f"arm joint {joint.value!r} is not calibrated") from None

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("arm controller is closed")
