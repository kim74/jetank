"""
Command-line entry point: ``jetank <command>``.

Layer: app
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jetank.app.wiring import build_camera
from jetank.config.settings import load_settings
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
        return _cmd_snapshot(settings, args.output, args.warmup)
    except JetankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
