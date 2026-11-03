"""
Command-line entry point: ``jetank <command>``.

Layer: app
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from jetank.app.signals import raise_on_termination_signals
from jetank.app.status_display import (
    close_display_quietly,
    run_status_display,
    show_status_until_interrupted,
)
from jetank.app.teleop_server import TeleopServer, load_page
from jetank.app.video_stream import VideoStream, encode_jpeg
from jetank.app.wiring import (
    build_camera,
    build_display,
    build_motor_driver,
    build_power_monitor,
    build_servo_bus,
)
from jetank.config.settings import load_settings
from jetank.control.arm import ArmController
from jetank.control.drive import DriveController
from jetank.control.gripper import GripperController
from jetank.control.status import StatusReporter
from jetank.control.teleop import TeleopSession
from jetank.domain.status import RobotMode
from jetank.drivers.system_clock import SystemClock
from jetank.drivers.system_host import SystemHostInfo
from jetank.errors import HardwareError, JetankError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jetank.config.settings import JetankSettings
    from jetank.domain.models import Frame
    from jetank.interfaces.camera import Camera
    from jetank.interfaces.display import Display
    from jetank.interfaces.power import PowerMonitor

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_FRAMES = 15
#: Extra wait after a gripper move's nominal time before its result is read back.
GRIPPER_SETTLE_S = 0.3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jetank", description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("config", help="print the resolved configuration as JSON")

    snapshot = commands.add_parser("snapshot", help="capture one camera frame to an image file")
    snapshot.add_argument("output", type=Path, help="image path, e.g. frame.jpg")
    snapshot.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="frames to discard first so auto-exposure can settle (default: %(default)s)",
    )

    commands.add_parser(
        "teleop", help="serve a remote-control web page (no authentication: trusted LAN only)"
    )
    commands.add_parser(
        "display", help="show hostname, IP address and battery on the OLED until Ctrl-C"
    )

    gripper = commands.add_parser(
        "gripper", help="read the gripper, or move it (then it goes limp again on exit)"
    )
    gripper.add_argument("action", choices=["status", "open", "close", "set"])
    gripper.add_argument(
        "opening", nargs="?", type=float, help="for 'set': 0.0 = closed ... 1.0 = open"
    )
    return parser


def _cmd_config(settings: JetankSettings) -> int:
    print(json.dumps(settings.model_dump(mode="json"), indent=2))
    return 0


def _cmd_snapshot(settings: JetankSettings, output: Path, warmup: int) -> int:
    if warmup < 0:
        raise JetankError("--warmup must not be negative")
    camera = build_camera(settings.camera, SystemClock())
    try:
        for _ in range(warmup):
            camera.read()
        frame = camera.read()
    finally:
        camera.close()
    _write_image(output, frame)
    print(f"saved {frame.width}x{frame.height} frame to {output}")
    return 0


def _cmd_teleop(settings: JetankSettings) -> int:
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
        display = _open_optional_display(settings)
        if display is not None:
            stack.callback(close_display_quietly, display)
        video = _open_optional_video(settings, clock, stack)
        servos = _open_optional_servos(settings, stack)
        gripper, arm = (None, None) if servos is None else servos
        motors = stack.enter_context(contextlib.closing(build_motor_driver(settings.motors)))
        drive = stack.enter_context(DriveController(motors, clock, settings.drive))
        stack.enter_context(raise_on_termination_signals())
        session = TeleopSession(drive, gripper, arm, settings.arm.jog_step)
        server = TeleopServer(session, settings.teleop, load_page(), video)
        reporter = None if display is None else _status_reporter(settings, display, power)
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


def _cmd_display(settings: JetankSettings) -> int:
    if settings.display.backend == "fake":
        logger.warning(
            "the display backend is FAKE: nothing will appear on the OLED "
            "(set display.backend: hardware, or JETANK_DISPLAY__BACKEND=hardware)"
        )
    with (
        contextlib.closing(build_power_monitor(settings.power)) as power,
        contextlib.closing(build_display(settings.display)) as display,
        raise_on_termination_signals(),
    ):
        print("showing status on the display (Ctrl-C stops and blanks it)")
        show_status_until_interrupted(
            _status_reporter(settings, display, power),
            RobotMode.IDLE,
            settings.display.refresh_period_s,
        )
    return 0


def _cmd_gripper(settings: JetankSettings, action: str, opening: float | None) -> int:
    if (action == "set") != (opening is not None):
        raise JetankError("an opening (0.0-1.0) is needed with 'set', and only with 'set'")
    if opening is not None and not 0.0 <= opening <= 1.0:
        raise JetankError(f"opening must be within 0.0-1.0, got {opening}")
    if action != "status" and settings.servos.backend == "fake":
        logger.warning("the servo backend is FAKE: the gripper will not move")
    config = settings.gripper
    with (
        contextlib.closing(build_servo_bus(settings.servos)) as bus,
        GripperController(
            bus, config.calibration, config.move_time_s, config.max_torque
        ) as gripper,
        raise_on_termination_signals(),
    ):
        if action != "status":
            target = {"open": 1.0, "close": 0.0}.get(action, opening)
            assert target is not None
            try:
                gripper.set_opening(target)
                time.sleep(config.move_time_s + GRIPPER_SETTLE_S)
            except KeyboardInterrupt as exc:
                logger.info("gripper move interrupted: %s", exc or "Ctrl-C")
        status = gripper.status()
    servo = status.servo
    print(
        f"gripper (servo {servo.servo_id}): opening {status.opening:.2f}, "
        f"position {servo.position}, {servo.voltage_v:.1f} V, {servo.temperature_c} °C, "
        f"torque {'on' if servo.torque_enabled else 'off'}"
        + (f", fault 0x{servo.error_flags:02X}" if servo.error_flags else "")
    )
    return 0


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


def _open_optional_display(settings: JetankSettings) -> Display | None:
    try:
        return build_display(settings.display)
    except HardwareError as exc:
        logger.warning("status display unavailable, continuing without it: %s", exc)
        return None


def _status_reporter(
    settings: JetankSettings, display: Display, power: PowerMonitor
) -> StatusReporter:
    return StatusReporter(display, power, SystemHostInfo(), settings.power.low_voltage_v)


def _write_image(path: Path, frame: Frame) -> None:
    import cv2

    try:
        written = cv2.imwrite(str(path), frame.image)
    except cv2.error as exc:
        raise JetankError(f"could not write image to {path}: {exc}") from exc
    if not written:
        raise JetankError(f"could not write image to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = load_settings(args.config)
        if args.command == "config":
            return _cmd_config(settings)
        if args.command == "teleop":
            return _cmd_teleop(settings)
        if args.command == "display":
            return _cmd_display(settings)
        if args.command == "gripper":
            return _cmd_gripper(settings, args.action, args.opening)
        return _cmd_snapshot(settings, args.output, args.warmup)
    except JetankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
