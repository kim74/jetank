from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from jetank.domain.servo import ArmPose, JointName, ServoState
from jetank.domain.teaching import (
    AnimalFit,
    GraspRecord,
    GraspResult,
    GripPoint,
    TeachLabels,
    TeachStage,
    ToyPosture,
    median_poses,
    record_id_for,
    summarize_fit,
)

T0 = datetime(2026, 11, 22, 10, 30, 5, 123456, tzinfo=timezone.utc)


def labels(
    animal: str = "lion",
    stage: TeachStage = TeachStage.GRASP,
    result: GraspResult = GraspResult.GOOD,
    posture: ToyPosture = ToyPosture.STANDING,
    grip_point: GripPoint = GripPoint.BODY,
) -> TeachLabels:
    return TeachLabels(
        animal=animal, stage=stage, posture=posture, grip_point=grip_point, result=result
    )


def state(servo_id: int, position: int) -> ServoState:
    return ServoState(
        servo_id=servo_id,
        position=position,
        voltage_v=5.4,
        temperature_c=30,
        moving=False,
        torque_enabled=True,
    )


def record(teach: TeachLabels, positions: dict[JointName, int], seconds: int = 0) -> GraspRecord:
    moment = T0 + timedelta(seconds=seconds)
    return GraspRecord(
        record_id=record_id_for(moment),
        recorded_at=moment,
        labels=teach,
        joints={joint: state(i, pos) for i, (joint, pos) in enumerate(positions.items(), 1)},
    )


def test_animal_names_are_trimmed_and_lower_case() -> None:
    assert labels(animal="  Lion ").animal == "lion"
    assert labels(animal="Löwe").animal == "löwe"
    assert labels(animal="baby-cow 2").animal == "baby-cow 2"


@pytest.mark.parametrize("animal", ["", "   ", "../etc", "cow/../x", "a" * 41, "-cow"])
def test_unusable_animal_names_are_rejected(animal: str) -> None:
    with pytest.raises(ValidationError):
        labels(animal=animal)


def test_note_length_is_limited() -> None:
    with pytest.raises(ValidationError):
        TeachLabels.model_validate({**labels().model_dump(), "note": "x" * 201})


def test_record_id_is_derived_from_the_time_with_microseconds() -> None:
    assert record_id_for(T0) == "20261122-103005-123456"


def test_record_needs_a_time_zone() -> None:
    with pytest.raises(ValidationError):
        GraspRecord(
            record_id=record_id_for(T0),
            recorded_at=T0.replace(tzinfo=None),
            labels=labels(),
            joints={},
        )


def test_record_rejects_an_id_that_could_name_another_file() -> None:
    with pytest.raises(ValidationError):
        GraspRecord(record_id="../x", recorded_at=T0, labels=labels(), joints={})


def test_record_pose_holds_the_measured_positions() -> None:
    taken = record(labels(), {JointName.BASE: 575, JointName.SHOULDER: 700})
    assert taken.pose == ArmPose(positions={JointName.BASE: 575, JointName.SHOULDER: 700})


def test_record_survives_a_json_round_trip() -> None:
    taken = record(labels(), {JointName.BASE: 575})
    assert GraspRecord.model_validate_json(taken.model_dump_json()) == taken


def test_fit_summary_counts_grasp_results_per_animal() -> None:
    pose = {JointName.BASE: 575}
    records = [
        record(labels("lion", result=GraspResult.GOOD, posture=ToyPosture.ON_SIDE), pose, 1),
        record(labels("lion", result=GraspResult.GOOD, posture=ToyPosture.ON_SIDE), pose, 2),
        record(labels("lion", result=GraspResult.SHAKY), pose, 3),
        record(labels("elephant", result=GraspResult.DOES_NOT_FIT), pose, 4),
        record(labels("cow", stage=TeachStage.CARRY), pose, 5),  # not a grasp record
    ]
    fits = summarize_fit(records)
    assert [fit.animal for fit in fits] == ["elephant", "lion"]
    elephant, lion = fits
    assert lion.results == {GraspResult.GOOD: 2, GraspResult.SHAKY: 1}
    assert lion.verdict is GraspResult.GOOD
    assert lion.best_posture is ToyPosture.ON_SIDE
    assert lion.best_grip_point is GripPoint.BODY
    assert elephant.verdict is GraspResult.DOES_NOT_FIT
    assert elephant.best_posture is None


def test_shaky_is_the_verdict_without_a_good_grasp() -> None:
    fit = AnimalFit(animal="sheep", results={GraspResult.SHAKY: 1, GraspResult.DOES_NOT_FIT: 2})
    assert fit.verdict is GraspResult.SHAKY


def test_median_pose_per_stage_uses_good_records_only() -> None:
    records = [
        record(labels(), {JointName.BASE: 570, JointName.SHOULDER: 700}, 1),
        record(labels(), {JointName.BASE: 580, JointName.SHOULDER: 710}, 2),
        record(labels(), {JointName.BASE: 900, JointName.SHOULDER: 705}, 3),  # outlier
        record(labels(result=GraspResult.SHAKY), {JointName.BASE: 100}, 4),
        record(labels(stage=TeachStage.SEARCH), {JointName.BASE: 575}, 5),
    ]
    poses = median_poses(records)
    assert poses == {
        TeachStage.GRASP: ArmPose(positions={JointName.BASE: 580, JointName.SHOULDER: 705}),
        TeachStage.SEARCH: ArmPose(positions={JointName.BASE: 575}),
    }


def test_median_pose_keeps_only_joints_every_record_measured() -> None:
    records = [
        record(labels(), {JointName.BASE: 570, JointName.ELBOW: 400}, 1),
        record(labels(), {JointName.BASE: 580}, 2),
    ]
    assert median_poses(records) == {TeachStage.GRASP: ArmPose(positions={JointName.BASE: 575})}


def test_no_good_records_give_no_poses() -> None:
    assert median_poses([record(labels(result=GraspResult.SHAKY), {JointName.BASE: 1})]) == {}
