"""
Arm: moves the arm joints' bus servos within their calibrated limits.

Layer: control

Safety rules:

* Every target is clamped to the joint's calibrated limits; joints without a calibration
  can never be commanded.
* Before a joint's torque is switched on, its goal is set to where it is now, so switching
  torque on cannot make it jump (a limp joint may rest outside its limits).
* Moves run at a bounded speed (``speed_cps``), however far the target.
* A jog (:meth:`ArmController.jog`) never sets a target more than ``JOG_MAX_LEAD_STEPS``
  steps ahead of where the servo is now, so the joint stops close by once jogging stops,
  even when a load holds it back.
* Only the configured joints' servos are ever touched.
* :meth:`ArmController.close` switches the torque of every engaged joint off: the arm goes
  limp and sags under gravity, so callers move it home first when they can, with
  :meth:`ArmController.park`. Parking moves only engaged joints; a limp joint is left resting.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from jetank.errors import ConfigurationError, DriverClosedError, ServoError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
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
#: Extra wait after a park move's nominal time before the joints are released.
PARK_SETTLE_S: Final = 0.3
#: How many jog steps a jog target may lead the servo's present position.
JOG_MAX_LEAD_STEPS: Final = 2


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
        self._targets: dict[JointName, int] = {}
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

    @property
    def targets(self) -> Mapping[JointName, int]:
        """Last commanded position of every engaged joint, in raw counts."""
        return dict(self._targets)

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
        duration_s = self._move_to(joint, present, target)
        logger.info("arm: %s %d -> %d in %.2f s", joint.value, present, target, duration_s)
        return duration_s

    def jog(self, joint: JointName, step: int) -> int:
        """Move ``joint`` ``step`` counts on from its last target, within its limits.

        Meant to be called repeatedly while an operator holds a key: each call extends the
        move a little, so the joint stops soon after the calls stop. The first jog of a
        joint starts from its present position. A target that has run ahead of the servo
        by more than ``JOG_MAX_LEAD_STEPS`` steps is pulled back first. At a limit the jog
        holds there silently.

        Args:
            joint: Joint to move.
            step: Signed change in raw counts (the direction depends on how the servo is
                mounted, see the joint calibration).

        Returns:
            The new target position in raw counts.

        Raises:
            ConfigurationError: ``joint`` has no calibration.
            ServoError: The servo did not answer or reported a fault.
        """
        self._ensure_open()
        cal = self._calibration(joint)
        present = self._bus.read_state(cal.servo_id).position
        lead = JOG_MAX_LEAD_STEPS * abs(step)
        start = max(present - lead, min(present + lead, self._targets.get(joint, present)))
        target = cal.clamp(start + step)
        self._move_to(joint, present, target)
        return target

    def home(self) -> float:
        """Start moving every joint to its home position. Does not wait.

        Returns:
            Duration of the longest of the moves, in seconds.
        """
        return max(
            (self.move(joint, cal.home_position) for joint, cal in self._joints.items()),
            default=0.0,
        )

    def park(self, sleep: Callable[[float], None]) -> None:
        """Move every engaged joint home, wait for it, then close (torque off). Idempotent.

        Meant for shutdown: the arm comes to rest in its home pose instead of falling from
        wherever it was held. Joints that are limp are not touched, so an arm nobody moved
        does not move. The joints are released even if homing fails or the wait is
        interrupted (e.g. by a second Ctrl-C).

        Args:
            sleep: Blocks for the given number of seconds (injected for tests).

        Raises:
            ServoError: A servo did not answer while homing or releasing.
        """
        if self._closed:
            return
        try:
            engaged = sorted(self._engaged, key=lambda j: j.value)
            if engaged:
                logger.info("arm: parking %s at home", ", ".join(j.value for j in engaged))
                duration_s = max(self.move(j, self._joints[j].home_position) for j in engaged)
                sleep(duration_s + PARK_SETTLE_S)
        finally:
            self.close()

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
                self._targets.pop(joint, None)
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

    def _move_to(self, joint: JointName, present: int, target: int) -> float:
        """Engage ``joint`` if needed and start the move; ``target`` is already clamped."""
        if joint not in self._engaged:
            self._engage(joint, present)
        duration_s = min(MAX_MOVE_S, max(MIN_MOVE_S, abs(target - present) / self._speed_cps))
        self._bus.move(self._joints[joint].servo_id, target, duration_s)
        self._targets[joint] = target
        return duration_s

    def _engage(self, joint: JointName, present: int) -> None:
        # Counted as engaged before any reply: a lost acknowledgement may hide a servo that
        # did switch its torque on, and close() must still switch it off.
        self._engaged.add(joint)
        servo_id = self._joints[joint].servo_id
        self._bus.move(servo_id, present, _PARK_S)
        self._targets[joint] = present
        self._bus.set_torque(servo_id, True)

    def _calibration(self, joint: JointName) -> JointCalibration:
        try:
            return self._joints[joint]
        except KeyError:
            raise ConfigurationError(f"arm joint {joint.value!r} is not calibrated") from None

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("arm controller is closed")
