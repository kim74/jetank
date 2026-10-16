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
from pathlib import Path
from typing import TYPE_CHECKING

from jetank.app.signals import raise_on_termination_signals
from jetank.app.teleop_server import TeleopServer, load_page
from jetank.app.wiring import build_camera, build_motor_driver
from jetank.config.settings import load_settings
from jetank.control.drive import DriveController
from jetank.control.teleop import TeleopSession
from jetank.drivers.system_clock import SystemClock
from jetank.errors import JetankError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jetank.config.settings import JetankSettings
    from jetank.domain.models import Frame

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_FRAMES = 15


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

    with (
        contextlib.closing(build_motor_driver(settings.motors)) as motors,
        DriveController(motors, clock, settings.drive) as drive,
        raise_on_termination_signals(),
    ):
        server = TeleopServer(TeleopSession(drive), settings.teleop, load_page())
        try:
            asyncio.run(server.serve(on_ready=announce))
        except KeyboardInterrupt as exc:
            logger.info("teleop stopped: %s", exc or "Ctrl-C")
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
            return _cmd_teleop(settings)
        return _cmd_snapshot(settings, args.output, args.warmup)
    except JetankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
