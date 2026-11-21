"""
Remote-control session: decides which connected client may drive and forwards its commands.

Layer: control

Network-agnostic: the server calls :meth:`TeleopSession.connect`, :meth:`~TeleopSession.handle`
and :meth:`~TeleopSession.disconnect` per client, and :meth:`~TeleopSession.tick` periodically.
Safety rules:

* Only the first client to connect drives; later clients are observers until it leaves.
* Any client may stop the robot (and ask for a fresh status with a ping).
* The driver disconnecting, or sending a malformed message, stops the robot.
* Only the driver may operate the gripper, jog the arm, move it to a named pose, put the robot
  to sleep, record a taught pose or record scenes. The servo commands are carried out by
  :class:`~jetank.control.teleop_servos.TeleopServos`; their faults are reported to the client
  but never stop the tracks, which are not needed to drive or stop safely.
* When the driver disconnects, the gripper is released (torque off) and automatic scene
  recording stops; the arm keeps holding where it is (going limp would let it fall). The
  caller parks it at home on shutdown (:meth:`~jetank.control.arm.ArmController.park`).
* If the driver goes quiet, the :class:`~jetank.control.drive.DriveController` watchdog stops
  the robot on the next :meth:`~TeleopSession.tick`.
* Sleep (a :class:`SleepMessage`, or a time without servo commands) stops the tracks (on
  request only, never when idle), stops automatic scene recording and lets the servos go limp,
  the arm after moving home. The next arm or gripper command wakes the robot.
* Recording a taught pose or scenes moves nothing. It counts as an arm command, as does every
  automatic scene, so the arm does not fall asleep (and the camera's viewpoint does not
  change) while poses are taught or scenes recorded. STOP does not end automatic recording.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.control.teleop_servos import TeleopServos
from jetank.domain.models import TrackSpeeds
from jetank.domain.status import RobotMode
from jetank.domain.teleop import (
    AnyClientMessage,
    ArmJogMessage,
    DriveMessage,
    GripperMessage,
    PingMessage,
    PoseMessage,
    Role,
    SleepMessage,
    StopMessage,
    TeachMessage,
    TeachReport,
    TeleopStatus,
)
from jetank.errors import HardwareError, ProtocolError, StorageError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jetank.control.arm import ArmController
    from jetank.control.drive import DriveController
    from jetank.control.gripper import GripperController
    from jetank.control.scene_recording import SceneRecorder
    from jetank.control.teaching import GraspTeacher
    from jetank.domain.scenes import SceneMessage
    from jetank.domain.servo import ArmPose
    from jetank.interfaces.clock import Clock

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
        clock: Time source for sleeping (when to release the arm, when it counts as idle).
        idle_sleep_s: Seconds without an arm or gripper command after which powered servos
            are put to sleep; 0 disables the automatic sleep.
        teacher: Records taught poses, or ``None`` to refuse :class:`TeachMessage`.
        poses: Named arm poses a :class:`PoseMessage` can move to.
        scenes: Records scenes, or ``None`` to refuse
            :class:`~jetank.domain.scenes.SceneMessage`.

    Raises:
        ValueError: ``jog_step`` is not positive or ``idle_sleep_s`` is negative.
    """

    def __init__(
        self,
        drive: DriveController,
        gripper: GripperController | None = None,
        arm: ArmController | None = None,
        jog_step: int = 15,
        *,
        clock: Clock,
        idle_sleep_s: float = 0.0,
        teacher: GraspTeacher | None = None,
        poses: Mapping[str, ArmPose] | None = None,
        scenes: SceneRecorder | None = None,
    ) -> None:
        self._servos = TeleopServos(
            gripper, arm, jog_step, poses or {}, clock=clock, idle_sleep_s=idle_sleep_s
        )
        self._teacher = teacher
        self._last_teach: TeachReport | None = None
        self._scenes = scenes
        self._drive = drive
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
            if self._scenes is not None:
                self._scenes.stop_auto()
            self._servos.driver_left()

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
        if isinstance(message, PingMessage):
            return self.status(client_id)
        if client_id != self._driver:
            return self.status(client_id, error="another client is driving")
        if isinstance(message, DriveMessage):
            self._speeds = self._drive.command(message.to_command())
            return self.status(client_id)
        return self.status(client_id, error=self._handle_driver_request(message))

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
        """Run the drive watchdog and the sleep timers.

        Returns:
            True if the watchdog stopped the robot just now.

        Raises:
            HardwareError: The motors failed. Servo failures are logged, not raised.
        """
        stopped = self._drive.tick()
        if self._scenes is not None:
            self._scenes.tick()
            if self._scenes.auto:
                self._servos.wake()  # keep the camera where it is while scenes are recorded
        self._servos.tick()
        return stopped

    def status(self, client_id: str, error: str | None = None) -> TeleopStatus:
        """Current state as seen by ``client_id``."""
        moving = self._drive.is_moving
        return TeleopStatus(
            role=Role.DRIVER if client_id == self._driver else Role.OBSERVER,
            moving=moving,
            speeds=self._speeds if moving else TrackSpeeds.stopped(),
            gripper=self._servos.gripper_report(),
            arm=self._servos.arm_report(),
            poses=self._servos.pose_names(),
            teach_enabled=self._teacher is not None,
            last_teach=self._last_teach,
            scenes=None if self._scenes is None else self._scenes.report(),
            asleep=self._servos.asleep,
            error=error,
        )

    def _handle_driver_request(
        self,
        message: GripperMessage
        | ArmJogMessage
        | PoseMessage
        | SleepMessage
        | TeachMessage
        | SceneMessage,
    ) -> str | None:
        """Carry out a driver's servo or recording request. Returns an error text."""
        if isinstance(message, SleepMessage):
            self._stop()
            if self._scenes is not None:
                self._scenes.stop_auto()
            return self._servos.fall_asleep()
        if isinstance(message, GripperMessage):
            return self._servos.operate_gripper(message.action, message.opening)
        if isinstance(message, ArmJogMessage):
            return self._servos.jog(message.joint, message.direction)
        if isinstance(message, PoseMessage):
            return self._servos.move_pose(message.name)
        self._servos.wake()
        if isinstance(message, TeachMessage):
            return self._teach(message)
        if self._scenes is None:
            return "scene recording is not available"
        return self._scenes.handle(message)

    def _teach(self, message: TeachMessage) -> str | None:
        """Record the present pose. Returns an error text instead of raising."""
        if self._teacher is None:
            return "teaching is not available"
        try:
            record = self._teacher.take(message.labels)
        except (HardwareError, StorageError) as exc:
            logger.warning("teleop: could not record the pose: %s", exc)
            return f"pose not recorded: {exc}"
        self._last_teach = TeachReport.from_record(record)
        return None

    def _stop(self) -> None:
        self._drive.stop()
        self._speeds = TrackSpeeds.stopped()

    def _require_connected(self, client_id: str) -> None:
        if client_id not in self._clients:
            raise ProtocolError(f"client {client_id} is not connected")
