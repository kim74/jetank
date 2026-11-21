"""
The ``jetank`` command line: commands, arguments and their help.

Layer: app
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from jetank.app.pose_review import DEFAULT_PORT as REVIEW_PORT
from jetank.app.prelabel_command import DEFAULT_DRAFT_CONFIDENCE
from jetank.app.scene_review import DEFAULT_PORT as SCENE_REVIEW_PORT

DESCRIPTION: Final = "Command-line entry point: ``jetank <command>``."
DEFAULT_WARMUP_FRAMES: Final = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jetank", description=DESCRIPTION)
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
        "'export' copies the images into the labelling folder; 'prelabel' lets the camera "
        "model draft drawings for the images not labelled yet",
    )
    scenes.add_argument(
        "action",
        nargs="?",
        choices=["summary", "review", "export", "prelabel"],
        default="summary",
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
        "--to",
        type=Path,
        help="'export', 'prelabel': labelling folder (default: dataset.labelling_directory)",
    )
    scenes.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_DRAFT_CONFIDENCE,
        help="'prelabel': draft boxes scoring at least this (default: %(default)s)",
    )
    analyze = commands.add_parser(
        "analyze",
        help="run the camera model (detector config) on saved images; moves nothing",
    )
    analyze.add_argument("images", nargs="+", type=Path, help="JPEG/PNG files, e.g. scenes")
    analyze.add_argument(
        "--output", type=Path, help="write the images with floor and detections drawn here"
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
