"""
Command-line entry point: ``jetank <command>``.

Layer: app
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from jetank.app.arm_commands import cmd_arm_status, cmd_poses
from jetank.app.dataset_commands import cmd_build_dataset, cmd_export_scenes
from jetank.app.pose_review import DEFAULT_PORT as REVIEW_PORT
from jetank.app.pose_review import cmd_review_poses
from jetank.app.scene_commands import cmd_scenes
from jetank.app.scene_review import DEFAULT_PORT as SCENE_REVIEW_PORT
from jetank.app.scene_review import cmd_review_scenes
from jetank.app.signals import raise_on_termination_signals
from jetank.app.status_display import (
    show_status_until_interrupted,
)
from jetank.app.teleop_command import cmd_teleop, status_reporter
from jetank.app.wiring import (
    build_camera,
    build_display,
    build_power_monitor,
    build_servo_bus,
)
from jetank.config.settings import load_settings
from jetank.control.gripper import GripperController
from jetank.domain.status import RobotMode
from jetank.drivers.system_clock import SystemClock
from jetank.errors import JetankError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jetank.config.settings import JetankSettings
    from jetank.domain.models import Frame

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

    arm = commands.add_parser("arm", help="read the arm joints and the gripper (moves nothing)")
    arm.add_argument("action", choices=["status"])

    poses = commands.add_parser(
        "poses",
        help="taught grasp records: 'summary' prints which animals fit and poses for the config; "
        "'review' serves a page to look through, relabel and delete them (no robot needed)",
    )
    poses.add_argument("action", nargs="?", choices=["summary", "review"], default="summary")
    poses.add_argument(
        "--directory", type=Path, help="records directory (default: teach.directory)"
    )
    poses.add_argument(
        "--host",
        default="0.0.0.0",  # reachable from the LAN on purpose, like teleop
        help="'review': interface to listen on (default: %(default)s; 127.0.0.1 = this machine)",
    )
    poses.add_argument(
        "--port", type=int, default=REVIEW_PORT, help="'review': TCP port (default: %(default)s)"
    )
    scenes = commands.add_parser(
        "scenes",
        help="recorded scenes: 'summary' counts them per room, light and with or without toy "
        "animals; 'review' serves a page to fix their labels and delete them (no robot needed); "
        "'export' copies the images into the labelling folder",
    )
    scenes.add_argument(
        "action", nargs="?", choices=["summary", "review", "export"], default="summary"
    )
    scenes.add_argument(
        "--directory", type=Path, help="scenes directory (default: scenes.directory)"
    )
    scenes.add_argument(
        "--host",
        default="0.0.0.0",  # reachable from the LAN on purpose, like teleop
        help="'review': interface to listen on (default: %(default)s; 127.0.0.1 = this machine)",
    )
    scenes.add_argument(
        "--port",
        type=int,
        default=SCENE_REVIEW_PORT,
        help="'review': TCP port (default: %(default)s)",
    )
    scenes.add_argument(
        "--to", type=Path, help="'export': labelling folder (default: dataset.labelling_directory)"
    )
    dataset = commands.add_parser(
        "dataset",
        help="'build': check the outlines drawn on the exported scenes and write the YOLO "
        "segmentation dataset for training",
    )
    dataset.add_argument("action", choices=["build"])
    dataset.add_argument("--scenes", type=Path, help="scenes directory (default: scenes.directory)")
    dataset.add_argument(
        "--labelling",
        type=Path,
        help="folder with the drawings (default: dataset.labelling_directory)",
    )
    dataset.add_argument(
        "--to", type=Path, help="dataset folder to write (default: dataset.directory)"
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
            status_reporter(settings, display, power),
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
            return cmd_teleop(settings)
        if args.command == "display":
            return _cmd_display(settings)
        if args.command == "gripper":
            return _cmd_gripper(settings, args.action, args.opening)
        if args.command == "arm":
            return cmd_arm_status(settings)
        if args.command == "poses" and args.action == "review":
            return cmd_review_poses(settings, args.directory, args.host, args.port)
        if args.command == "poses":
            return cmd_poses(settings, args.directory)
        if args.command == "scenes" and args.action == "export":
            return cmd_export_scenes(settings, args.directory, args.to)
        if args.command == "dataset":
            return cmd_build_dataset(settings, args.scenes, args.labelling, args.to)
        if args.command == "scenes" and args.action == "review":
            return cmd_review_scenes(settings, args.directory, args.host, args.port)
        if args.command == "scenes":
            return cmd_scenes(settings, args.directory)
        return _cmd_snapshot(settings, args.output, args.warmup)
    except JetankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
