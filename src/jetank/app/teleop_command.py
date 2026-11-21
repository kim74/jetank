"""
``jetank teleop``: wires the robot into the remote-control web server and runs it.

Layer: app

Every part except the motors is optional: without a display, servo bus or camera, teleop
runs without what they provide (warning once). On exit the motors stop first, then the arm
parks at home and goes limp, then the gripper goes limp, then the display is blanked.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import time
from datetime import datetime
from typing import TYPE_CHECKING

from jetank.app.signals import raise_on_termination_signals
from jetank.app.status_display import close_display_quietly, run_status_display
from jetank.app.teleop_server import TeleopServer, load_page
from jetank.app.video_stream import VideoStream, encode_jpeg
from jetank.app.wiring import (
    build_camera,
    build_display,
    build_motor_driver,
    build_power_monitor,
    build_servo_bus,
)
from jetank.control.arm import ArmController
from jetank.control.drive import DriveController
from jetank.control.gripper import GripperController
from jetank.control.scene_recording import SceneRecorder
from jetank.control.status import StatusReporter
from jetank.control.teaching import GraspTeacher
from jetank.control.teleop import TeleopSession
from jetank.drivers.grasp_directory import GraspDirectory
from jetank.drivers.scene_directory import SceneDirectory
from jetank.drivers.system_clock import SystemClock
from jetank.drivers.system_host import SystemHostInfo
from jetank.errors import HardwareError

if TYPE_CHECKING:
    from collections.abc import Callable

    from jetank.config.settings import JetankSettings
    from jetank.interfaces.camera import Camera
    from jetank.interfaces.display import Display
    from jetank.interfaces.power import PowerMonitor

logger = logging.getLogger(__name__)


def cmd_teleop(settings: JetankSettings) -> int:
    """Serve the remote-control page until Ctrl-C or SIGTERM. Returns the exit code."""
    clock = SystemClock()
    if settings.motors.backend == "fake":
        logger.warning(
            "teleop is using FAKE motors: the robot will not move "
            "(set motors.backend: hardware, or JETANK_MOTORS__BACKEND=hardware)"
        )
    else:
        logger.info("teleop is using the %s motor backend", settings.motors.backend)

    def announce(port: int) -> None:
        print(f"remote control at http://{socket.gethostname()}.local:{port}/ (Ctrl-C stops)")

    # Entered in this order so that on exit the motors stop first, then the arm parks at
    # home and goes limp, then the gripper goes limp, then the display is blanked.
    with contextlib.ExitStack() as stack:
        power = stack.enter_context(contextlib.closing(build_power_monitor(settings.power)))
        display = open_optional_display(settings)
        if display is not None:
            stack.callback(close_display_quietly, display)
        video = _open_optional_video(settings, clock, stack)
        servos = _open_optional_servos(settings, stack)
        gripper, arm = (None, None) if servos is None else servos
        teacher = None if arm is None else _teacher(settings, arm, gripper, video)
        scenes = None if video is None else _scene_recorder(settings, arm, video, clock)
        motors = stack.enter_context(contextlib.closing(build_motor_driver(settings.motors)))
        drive = stack.enter_context(DriveController(motors, clock, settings.drive))
        stack.enter_context(raise_on_termination_signals())
        session = TeleopSession(
            drive,
            gripper,
            arm,
            settings.arm.jog_step,
            clock=clock,
            idle_sleep_s=settings.teleop.idle_sleep_s,
            teacher=teacher,
            poses=settings.arm.poses,
            scenes=scenes,
        )
        server = TeleopServer(session, settings.teleop, load_page(), video)
        reporter = None if display is None else status_reporter(settings, display, power)
        try:
            asyncio.run(_serve_teleop(server, announce, reporter, session, settings))
        except KeyboardInterrupt as exc:
            logger.info("teleop stopped: %s", exc or "Ctrl-C")
    return 0


async def _serve_teleop(
    server: TeleopServer,
    announce: Callable[[int], None],
    reporter: StatusReporter | None,
    session: TeleopSession,
    settings: JetankSettings,
) -> None:
    if reporter is None:
        await server.serve(on_ready=announce)
        return
    status = asyncio.create_task(
        run_status_display(reporter, lambda: session.mode, settings.display.refresh_period_s)
    )
    try:
        await server.serve(on_ready=announce)
    finally:
        status.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await status


def _open_optional_servos(
    settings: JetankSettings, stack: contextlib.ExitStack
) -> tuple[GripperController, ArmController] | None:
    """Gripper and arm for teleop, or ``None`` (with a warning) if the servo bus is unavailable.

    On exit the arm first moves the joints it holds home and goes limp there, then the gripper
    goes limp, then the bus is closed.
    """
    try:
        bus = build_servo_bus(settings.servos)
    except HardwareError as exc:
        logger.warning("servo bus unavailable, teleop continues without gripper and arm: %s", exc)
        return None
    stack.callback(bus.close)
    config = settings.gripper
    gripper = GripperController(bus, config.calibration, config.move_time_s, config.max_torque)
    stack.callback(gripper.close)
    arm = ArmController(bus, settings.arm.joints, settings.arm.speed_cps)
    stack.callback(arm.park, time.sleep)
    if settings.servos.backend == "fake":
        logger.warning("teleop uses the FAKE servo bus: gripper and arm will not move")
    return gripper, arm


def _teacher(
    settings: JetankSettings,
    arm: ArmController,
    gripper: GripperController | None,
    video: VideoStream | None,
) -> GraspTeacher:
    """Records poses from the teleop page, with the frame the page's video shows."""

    def snapshot() -> bytes | None:
        return None if video is None else video.latest_jpeg

    return GraspTeacher(
        arm,
        gripper,
        GraspDirectory(settings.teach.directory),
        snapshot,
        lambda: datetime.now().astimezone(),
        settings.gripper.holding_margin,
    )


def _scene_recorder(
    settings: JetankSettings,
    arm: ArmController | None,
    video: VideoStream,
    clock: SystemClock,
) -> SceneRecorder:
    """Records scenes from the frame the page's video shows, sharper than the video itself."""
    quality = settings.scenes.jpeg_quality

    def snapshot() -> bytes | None:
        frame = video.latest_frame
        return None if frame is None else encode_jpeg(frame, quality)

    return SceneRecorder(
        SceneDirectory(settings.scenes.directory),
        snapshot,
        arm,
        now=lambda: datetime.now().astimezone(),
        clock=clock,
        auto_period_s=settings.scenes.auto_period_s,
        auto_limit_s=settings.scenes.auto_limit_s,
    )


def _open_optional_video(
    settings: JetankSettings, clock: SystemClock, stack: contextlib.ExitStack
) -> VideoStream | None:
    """Live video for teleop, or ``None`` (with a warning) if the camera cannot be opened."""
    try:
        camera: Camera = build_camera(settings.camera, clock)
    except HardwareError as exc:
        logger.warning("camera unavailable, teleop continues without video: %s", exc)
        return None
    stack.callback(camera.close)
    if settings.camera.backend == "fake":
        logger.warning("teleop uses the FAKE camera: the video shows a test pattern")
    quality = settings.teleop.video_jpeg_quality
    video = VideoStream(
        camera, settings.teleop.video_fps, lambda frame: encode_jpeg(frame, quality)
    )
    stack.callback(video.close)
    return video


def open_optional_display(settings: JetankSettings) -> Display | None:
    try:
        return build_display(settings.display)
    except HardwareError as exc:
        logger.warning("status display unavailable, continuing without it: %s", exc)
        return None


def status_reporter(
    settings: JetankSettings, display: Display, power: PowerMonitor
) -> StatusReporter:
    return StatusReporter(display, power, SystemHostInfo(), settings.power.low_voltage_v)
