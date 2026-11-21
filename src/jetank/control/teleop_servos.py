"""
The servo side of remote control: gripper, arm jogs, named poses and sleep.

Layer: control

:class:`~jetank.control.teleop.TeleopSession` decides *who* may command the servos; this class
carries the commands out. Every method reports a failure as an error text instead of raising:
the gripper and arm are not needed to drive or stop safely, so their faults never stop the
tracks.

* Each jog moves a joint one small step, so a joint stops soon after the jog messages stop.
* A pose moves its joints at the arm's bounded speed and keeps holding there.
* Sleep (on request, or ``idle_sleep_s`` without a servo command while one is powered): the
  gripper goes limp, and the arm moves home and goes limp there on a later :meth:`tick`
  (:class:`~jetank.control.servo_sleep.ServoSleep`). An arm that cannot move home keeps holding.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.control.servo_sleep import ServoSleep
from jetank.domain.teleop import ArmReport, GripperAction, GripperReport
from jetank.errors import ConfigurationError, HardwareError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jetank.control.arm import ArmController
    from jetank.control.gripper import GripperController
    from jetank.domain.servo import ArmPose, JointName
    from jetank.interfaces.clock import Clock

logger = logging.getLogger(__name__)


class TeleopServos:
    """Carries out gripper and arm commands for a teleop session. Controllers are borrowed.

    Not thread-safe: call it from a single thread or event loop.

    Args:
        gripper: Gripper, or ``None`` if the robot has none (or it is unavailable).
        arm: Arm, or ``None`` if the robot has none (or it is unavailable).
        jog_step: Raw counts an arm joint moves per jog.
        poses: Named arm poses that :meth:`move_pose` can move to.
        clock: Time source for sleeping.
        idle_sleep_s: Seconds without a servo command after which powered servos are put to
            sleep; 0 disables the automatic sleep.

    Raises:
        ValueError: ``jog_step`` is not positive or ``idle_sleep_s`` is negative.
    """

    def __init__(
        self,
        gripper: GripperController | None,
        arm: ArmController | None,
        jog_step: int,
        poses: Mapping[str, ArmPose],
        *,
        clock: Clock,
        idle_sleep_s: float,
    ) -> None:
        if jog_step < 1:
            raise ValueError("jog step must be positive")
        self._gripper = gripper
        self._arm = arm
        self._jog_step = jog_step
        self._poses = dict(poses)
        self._sleep = ServoSleep(arm, clock, idle_sleep_s)
        self._gripper_report = GripperReport(engaged=False)

    @property
    def asleep(self) -> bool:
        """Whether the servos were put to sleep and no command has woken them since."""
        return self._sleep.asleep

    def gripper_report(self) -> GripperReport | None:
        """Gripper state as last commanded, or ``None`` without a gripper."""
        return None if self._gripper is None else self._gripper_report

    def arm_report(self) -> ArmReport | None:
        """Arm state as last commanded, or ``None`` without an arm."""
        return None if self._arm is None else ArmReport(targets=dict(self._arm.targets))

    def pose_names(self) -> list[str]:
        """Names of the poses :meth:`move_pose` can move to; none without an arm."""
        return [] if self._arm is None else sorted(self._poses)

    def wake(self) -> None:
        """A servo command (or something that needs the arm where it is) arrived: stay awake."""
        self._sleep.wake()

    def operate_gripper(self, action: GripperAction, opening: float | None) -> str | None:
        """Wake up and carry out a gripper action. Returns an error text instead of raising."""
        self._sleep.wake()
        return self._operate_gripper(action, opening)

    def jog(self, joint: JointName, direction: int) -> str | None:
        """Wake up and move one joint a step. Returns an error text instead of raising."""
        self._sleep.wake()
        if self._arm is None:
            return "this robot has no arm"
        try:
            self._arm.jog(joint, direction * self._jog_step)
        except ConfigurationError as exc:
            return str(exc)
        except HardwareError as exc:
            logger.warning("teleop: arm failed: %s", exc)
            return f"arm failed: {exc}"
        return None

    def move_pose(self, name: str) -> str | None:
        """Wake up and start moving the arm to a named pose. Returns an error text."""
        self._sleep.wake()
        if self._arm is None:
            return "this robot has no arm"
        pose = self._poses.get(name)
        if pose is None:
            return f"no arm pose {name!r} is configured"
        try:
            self._arm.move_pose(pose)
        except ConfigurationError as exc:
            return str(exc)
        except HardwareError as exc:
            logger.warning("teleop: arm failed moving to %s: %s", name, exc)
            return f"arm failed: {exc}"
        logger.info("teleop: arm moving to pose %s", name)
        return None

    def driver_left(self) -> None:
        """The driver disconnected: release the gripper; the arm keeps holding where it is."""
        if self._gripper is not None:
            # Also after a failed command, when the gripper's torque state is unknown.
            self._operate_gripper(GripperAction.RELEASE, None)

    def fall_asleep(self) -> str | None:
        """Let the gripper go limp and start the arm's way home. Returns an error text."""
        gripper_error = None
        if self._gripper is not None:  # also when its state is unknown after a failure
            gripper_error = self._operate_gripper(GripperAction.RELEASE, None)
        return self._sleep.fall_asleep() or gripper_error

    def tick(self) -> None:
        """Run the sleep timers: release a homed arm, fall asleep when idle. Never raises."""
        if self._sleep.tick(self._powered()):
            logger.info(
                "teleop: no arm or gripper command for %.0f s, going to sleep",
                self._sleep.idle_sleep_s,
            )
            self.fall_asleep()

    def _operate_gripper(self, action: GripperAction, opening: float | None) -> str | None:
        if self._gripper is None:
            return "this robot has no gripper"
        try:
            if action is GripperAction.RELEASE:
                self._gripper.release()
                self._gripper_report = GripperReport(engaged=False)
                return None
            target = {GripperAction.OPEN: 1.0, GripperAction.CLOSE: 0.0}.get(action, opening)
            assert target is not None  # GripperMessage guarantees an opening with SET
            self._gripper.set_opening(target)
        except HardwareError as exc:
            logger.warning("teleop: gripper failed: %s", exc)
            # Its state is unknown now; report it as limp and let the next command retry.
            self._gripper_report = GripperReport(engaged=False)
            return f"gripper failed: {exc}"
        self._gripper_report = GripperReport(engaged=True, opening=target)
        return None

    def _powered(self) -> bool:
        arm_powered = self._arm is not None and self._arm.engaged
        return arm_powered or self._gripper_report.engaged
