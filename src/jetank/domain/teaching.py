"""
Grasp teaching: what an operator records while posing the arm around a toy by hand.

Layer: domain

A :class:`GraspRecord` holds the operator's labels (which animal, which stage of a grasp, how
the toy lies, where it is gripped, how well it worked) together with what the robot measured
at that moment (every joint, the gripper). Records are summarised into which animals fit the
gripper (:func:`summarize_fit`) and into one pose per stage for the configuration
(:func:`median_poses`).
"""

from __future__ import annotations

import statistics
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Final, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from jetank.domain.servo import ArmPose, JointName, ServoPosition, ServoState

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

_T = TypeVar("_T")

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")

#: Longest note an operator can attach to a record.
MAX_NOTE_LENGTH: Final = 200

#: Name of a toy, e.g. "lion" or "löwe": letters, digits, spaces, "-" and "_", lower case.
AnimalName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, min_length=1, max_length=40, pattern=r"^\w[\w \-]*$"
    ),
]
#: Identifier of a record; also its file name stem, so it is restricted to safe characters.
RecordId = Annotated[str, StringConstraints(pattern=r"^[0-9]{8}-[0-9]{6}-[0-9]{6}$")]


class TeachStage(str, Enum):
    """Step of a grasp that a pose belongs to; each stage becomes one configured pose."""

    SEARCH = "search"  # driving and looking for toys: camera on the floor ahead
    PRE_GRASP = "pre_grasp"  # jaws open just above the toy
    GRASP = "grasp"  # jaws around the toy
    CARRY = "carry"  # toy lifted, camera still sees the way ahead
    PLACE = "place"  # toy set down at the drop-off


class ToyPosture(str, Enum):
    """How the toy lies on the floor."""

    STANDING = "standing"
    ON_SIDE = "on_side"
    ON_BACK = "on_back"


class GripPoint(str, Enum):
    """Which part of the toy the jaws hold."""

    BODY = "body"
    LEGS = "legs"
    HEAD = "head"
    NECK = "neck"


class GraspResult(str, Enum):
    """The operator's verdict on the pose."""

    GOOD = "good"
    SHAKY = "shaky"
    DOES_NOT_FIT = "does_not_fit"


class TeachLabels(BaseModel):
    """What the operator says about a pose.

    Attributes:
        animal: Which toy, e.g. "lion".
        stage: Step of the grasp the pose belongs to.
        posture: How the toy lies.
        grip_point: Where the jaws hold it.
        result: How well it worked.
        note: Free text, at most ``MAX_NOTE_LENGTH`` characters.
    """

    model_config = _VALUE_OBJECT

    animal: AnimalName
    stage: TeachStage
    posture: ToyPosture
    grip_point: GripPoint
    result: GraspResult
    note: str = Field(default="", max_length=MAX_NOTE_LENGTH)


class GripperReading(BaseModel):
    """The gripper at the moment a pose was taken.

    Attributes:
        servo: Live servo reading.
        target: Last position commanded while powered, or ``None`` when limp.
        holding: Whether the jaws were told to close and stopped well short of the closing
            target, i.e. something lies between them.
    """

    model_config = _VALUE_OBJECT

    servo: ServoState
    target: ServoPosition | None
    holding: bool


class GraspRecord(BaseModel):
    """One taught pose: the operator's labels and what the robot measured.

    Attributes:
        record_id: Unique ID, derived from the time the pose was taken.
        recorded_at: Wall-clock time the pose was taken, with time zone.
        labels: What the operator said about it.
        joints: Live reading of every calibrated arm joint.
        gripper: Gripper reading, or ``None`` when the robot has no gripper.
        image_file: Name of the camera image saved with the record, if there was one.
    """

    model_config = _VALUE_OBJECT

    record_id: RecordId
    recorded_at: AwareDatetime
    labels: TeachLabels
    joints: dict[JointName, ServoState]
    gripper: GripperReading | None = None
    image_file: str | None = None

    @property
    def pose(self) -> ArmPose:
        """The measured joint positions as a pose."""
        return ArmPose(positions={joint: state.position for joint, state in self.joints.items()})


def record_id_for(moment: datetime) -> str:
    """The record ID for a pose taken at ``moment`` (microseconds keep IDs apart)."""
    return moment.strftime("%Y%m%d-%H%M%S-%f")


class AnimalFit(BaseModel):
    """How well one toy fits the gripper, over all its grasp records.

    Attributes:
        animal: Which toy.
        results: Number of grasp-stage records per result.
        best_posture / best_grip_point: Most frequent posture and grip point among the
            ``good`` grasps, or ``None`` if there is no good grasp.
    """

    model_config = _VALUE_OBJECT

    animal: str
    results: dict[GraspResult, int]
    best_posture: ToyPosture | None = None
    best_grip_point: GripPoint | None = None

    @property
    def verdict(self) -> GraspResult:
        """Overall verdict: good if any grasp was good, else shaky if any was, else no fit."""
        for result in (GraspResult.GOOD, GraspResult.SHAKY):
            if self.results.get(result, 0) > 0:
                return result
        return GraspResult.DOES_NOT_FIT


def summarize_fit(records: Iterable[GraspRecord]) -> list[AnimalFit]:
    """Which animals fit the gripper, from the records of the ``grasp`` stage.

    Returns:
        One entry per animal with grasp records, sorted by name.
    """
    grasps: dict[str, list[TeachLabels]] = {}
    for record in records:
        if record.labels.stage is TeachStage.GRASP:
            grasps.setdefault(record.labels.animal, []).append(record.labels)
    fits = []
    for animal in sorted(grasps):
        labels = grasps[animal]
        good = [lab for lab in labels if lab.result is GraspResult.GOOD]
        fits.append(
            AnimalFit(
                animal=animal,
                results=dict(Counter(lab.result for lab in labels)),
                best_posture=_most_common(lab.posture for lab in good),
                best_grip_point=_most_common(lab.grip_point for lab in good),
            )
        )
    return fits


def median_poses(records: Iterable[GraspRecord]) -> dict[TeachStage, ArmPose]:
    """One pose per stage: the median of every joint over the ``good`` records of that stage.

    The median ignores a few outliers (a joint nudged by accident) better than the mean.
    A joint enters a stage's pose only if every good record of that stage measured it.

    Returns:
        A pose for each stage with at least one good record.
    """
    poses_by_stage: dict[TeachStage, list[ArmPose]] = {}
    for record in records:
        if record.labels.result is GraspResult.GOOD:
            poses_by_stage.setdefault(record.labels.stage, []).append(record.pose)
    medians = {}
    for stage, poses in poses_by_stage.items():
        joints = set.intersection(*(set(pose.positions) for pose in poses))
        if joints:
            medians[stage] = ArmPose(
                positions={
                    joint: round(statistics.median(pose.positions[joint] for pose in poses))
                    for joint in sorted(joints, key=lambda j: j.value)
                }
            )
    return medians


def _most_common(values: Iterable[_T]) -> _T | None:
    counts = Counter(values).most_common(1)
    return counts[0][0] if counts else None
