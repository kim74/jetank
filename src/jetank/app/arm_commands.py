"""
CLI commands around the arm that never move it: ``jetank arm status`` and ``jetank poses``.

Layer: app
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from jetank.app.wiring import build_servo_bus
from jetank.control.arm import ArmController
from jetank.control.gripper import GripperController
from jetank.domain.teaching import GraspResult, median_poses, summarize_fit
from jetank.drivers.grasp_directory import GraspDirectory

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.domain.servo import ServoState
    from jetank.domain.teaching import AnimalFit, GraspRecord

_VERDICT = {
    GraspResult.GOOD: "fits",
    GraspResult.SHAKY: "fits, but shaky",
    GraspResult.DOES_NOT_FIT: "does not fit",
}


def cmd_arm_status(settings: JetankSettings) -> int:
    """Print every arm joint and the gripper, read from the bus. Moves nothing."""
    config = settings.gripper
    with (
        contextlib.closing(build_servo_bus(settings.servos)) as bus,
        ArmController(bus, settings.arm.joints, settings.arm.speed_cps) as arm,
        GripperController(
            bus, config.calibration, config.move_time_s, config.max_torque
        ) as gripper,
    ):
        for joint, state in arm.read_pose().items():
            cal = settings.arm.joints[joint]
            print(
                f"{joint.value:<12} {_servo_line(state)}  "
                f"limits {cal.min_position}..{cal.max_position}  home {cal.home_position}"
            )
        status = gripper.status()
        print(f"{'gripper':<12} {_servo_line(status.servo)}  opening {status.opening:.2f}")
    return 0


def cmd_poses(settings: JetankSettings, directory: Path | None) -> int:
    """Summarise the taught grasp records: which animals fit, and one pose per stage."""
    source = directory if directory is not None else settings.teach.directory
    records = GraspDirectory(source).load_all()
    if not records:
        print(f"no grasp records in {source}")
        return 0
    print(f"{len(records)} grasp records in {source}\n")
    for line in format_fit(summarize_fit(records)):
        print(line)
    print()
    for line in format_poses(records):
        print(line)
    return 0


def format_fit(fits: list[AnimalFit]) -> list[str]:
    """Table of which animals fit the gripper, from the grasp-stage records."""
    if not fits:
        return ["no records of the grasp stage yet: no verdict on which animals fit"]
    lines = ["Animals (records of the grasp stage):"]
    width = max(len(fit.animal) for fit in fits)
    for fit in fits:
        counts = "  ".join(f"{r.value} {fit.results.get(r, 0)}" for r in GraspResult)
        best = ""
        if fit.best_posture is not None and fit.best_grip_point is not None:
            best = f" ({fit.best_posture.value}, gripped at {fit.best_grip_point.value})"
        lines.append(f"  {fit.animal:<{width}}  {counts}  -> {_VERDICT[fit.verdict]}{best}")
    return lines


def format_poses(records: list[GraspRecord]) -> list[str]:
    """YAML snippet with the median pose of every stage, ready to paste into the config."""
    poses = median_poses(records)
    if not poses:
        return ["no 'good' records yet: no poses to suggest"]
    counts: dict[str, int] = {}
    for record in records:
        if record.labels.result is GraspResult.GOOD:
            counts[record.labels.stage.value] = counts.get(record.labels.stage.value, 0) + 1
    lines = ["Poses (median of the 'good' records), to paste under `arm:` in config/jetank.yaml:"]
    lines.append("  poses:")
    for stage, pose in sorted(poses.items(), key=lambda item: item[0].value):
        joints = ", ".join(f"{joint.value}: {pos}" for joint, pos in pose.positions.items())
        lines.append(
            f"    {stage.value}: {{positions: {{{joints}}}}}  # from {counts[stage.value]} records"
        )
    return lines


def _servo_line(state: ServoState) -> str:
    return (
        f"(servo {state.servo_id}): position {state.position:>4}  {state.temperature_c} °C  "
        f"torque {'on ' if state.torque_enabled else 'off'}"
    )
