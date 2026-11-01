"""
Remote-control session: decides which connected client may drive and forwards its commands.

Layer: control

Network-agnostic: the server calls :meth:`TeleopSession.connect`, :meth:`~TeleopSession.handle`
and :meth:`~TeleopSession.disconnect` per client, and :meth:`~TeleopSession.tick` periodically.
Safety rules:

* Only the first client to connect drives; later clients are observers until it leaves.
* Any client may stop the robot.
* The driver disconnecting, or sending a malformed message, stops the robot.
* Only the driver may operate the gripper. When the driver disconnects, the gripper is
  released (torque off). A gripper fault is reported to the client but does not stop
  the tracks: the gripper is not needed to drive or stop safely.
* Only the driver may jog the arm. Each jog moves a joint one small step, so a joint stops
  soon after the jog messages stop. When the driver disconnects, the arm keeps holding
  where it is (going limp would let it fall). An arm fault is reported like a gripper
  fault and does not stop the tracks.
* If the driver goes quiet, the :class:`~jetank.control.drive.DriveController` watchdog stops
  the robot on the next :meth:`~TeleopSession.tick`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.domain.models import TrackSpeeds
from jetank.domain.status import RobotMode
from jetank.domain.teleop import (
    AnyClientMessage,
    ArmJogMessage,
    ArmReport,
    GripperAction,
    GripperMessage,
    GripperReport,
    Role,
    StopMessage,
    TeleopStatus,
)
from jetank.errors import ConfigurationError, HardwareError, ProtocolError

if TYPE_CHECKING:
    from jetank.control.arm import ArmController
    from jetank.control.drive import DriveController
    from jetank.control.gripper import GripperController

logger = logging.getLogger(__name__)


class TeleopSession:
    """Arbitrates remote clients in front of one :class:`DriveController`.

    Not thread-safe: call it from a single thread or event loop. The controllers are
    borrowed, not owned; the caller stops and closes the hardware.

    Args:
        drive: Track drive.
        gripper: Gripper, or ``None`` if the robot has none (or it is unavailable).
        arm: Arm, or ``None`` if the robot has none (or it is unavailable).
        jog_step: Raw counts an arm joint moves per :class:`ArmJogMessage`.

    Raises:
        ValueError: ``jog_step`` is not positive.
    """

    def __init__(
        self,
        drive: DriveController,
        gripper: GripperController | None = None,
        arm: ArmController | None = None,
        jog_step: int = 15,
    ) -> None:
        if jog_step < 1:
            raise ValueError("jog step must be positive")
        self._drive = drive
        self._gripper = gripper
        self._arm = arm
        self._jog_step = jog_step
        self._gripper_report = GripperReport(engaged=False)
        self._clients: set[str] = set()
        self._driver: str | None = None
        self._speeds = TrackSpeeds.stopped()

    @property
    def driver(self) -> str | None:
        """ID of the client currently allowed to drive, if any."""
        return self._driver

    @property
    def mode(self) -> RobotMode:
        """Whether a client is in the driver seat, for the status display."""
        return RobotMode.TELEOP_WAITING if self._driver is None else RobotMode.TELEOP_DRIVING

    def connect(self, client_id: str) -> TeleopStatus:
        """Register a client. It becomes the driver if nobody else is driving.

        Raises:
            ProtocolError: A client with this ID is already connected.
        """
        if client_id in self._clients:
            raise ProtocolError(f"client {client_id} is already connected")
        self._clients.add(client_id)
        if self._driver is None:
            self._driver = client_id
            logger.info("teleop: %s is now driving", client_id)
        else:
            logger.info("teleop: %s connected as observer (%s is driving)", client_id, self._driver)
        return self.status(client_id)

    def disconnect(self, client_id: str) -> None:
        """Forget a client. If it was driving, the robot stops and the driver seat is free."""
        self._clients.discard(client_id)
        if client_id == self._driver:
            self._driver = None
            self._stop()
            logger.info("teleop: driver %s disconnected, robot stopped", client_id)
            if self._gripper is not None:
                # Also after a failed command, when the gripper's torque state is unknown.
                self._operate_gripper(GripperAction.RELEASE, None)

    def handle(self, client_id: str, message: AnyClientMessage) -> TeleopStatus:
        """Apply one message from a client and report the resulting state.

        Raises:
            ProtocolError: The client is not connected.
            HardwareError: The motors failed (the controller has already tried to stop them).
        """
        self._require_connected(client_id)
        if isinstance(message, StopMessage):
            self._stop()
            return self.status(client_id)
        if client_id != self._driver:
            return self.status(client_id, error="another client is driving")
        if isinstance(message, GripperMessage):
            error = self._operate_gripper(message.action, message.opening)
            return self.status(client_id, error=error)
        if isinstance(message, ArmJogMessage):
            return self.status(client_id, error=self._jog_arm(message))
        self._speeds = self._drive.command(message.to_command())
        return self.status(client_id)

    def reject(self, client_id: str, error: str) -> TeleopStatus:
        """Report an unusable message. If it came from the driver, stop to be safe.

        Raises:
            ProtocolError: The client is not connected.
        """
        self._require_connected(client_id)
        if client_id == self._driver:
            self._stop()
        return self.status(client_id, error=error)

    def tick(self) -> bool:
        """Run the drive watchdog. Returns True if it stopped the robot just now."""
        return self._drive.tick()

    def status(self, client_id: str, error: str | None = None) -> TeleopStatus:
        """Current state as seen by ``client_id``."""
        moving = self._drive.is_moving
        return TeleopStatus(
            role=Role.DRIVER if client_id == self._driver else Role.OBSERVER,
            moving=moving,
            speeds=self._speeds if moving else TrackSpeeds.stopped(),
            gripper=None if self._gripper is None else self._gripper_report,
            arm=None if self._arm is None else ArmReport(targets=dict(self._arm.targets)),
            error=error,
        )

    def _operate_gripper(self, action: GripperAction, opening: float | None) -> str | None:
        """Carry out a gripper action. Returns an error text instead of raising."""
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

    def _jog_arm(self, message: ArmJogMessage) -> str | None:
        """Move one joint a step. Returns an error text instead of raising."""
        if self._arm is None:
            return "this robot has no arm"
        try:
            self._arm.jog(message.joint, message.direction * self._jog_step)
        except ConfigurationError as exc:
            return str(exc)
        except HardwareError as exc:
            logger.warning("teleop: arm failed: %s", exc)
            return f"arm failed: {exc}"
        return None

    def _stop(self) -> None:
        self._drive.stop()
        self._speeds = TrackSpeeds.stopped()

    def _require_connected(self, client_id: str) -> None:
        if client_id not in self._clients:
            raise ProtocolError(f"client {client_id} is not connected")
