from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jetank.control.arm import ArmController
from jetank.control.gripper import GripperController
from jetank.control.teaching import GraspTeacher
from jetank.domain.servo import GripperCalibration, JointCalibration, JointName
from jetank.domain.teaching import GraspResult, GripPoint, TeachLabels, TeachStage, ToyPosture
from jetank.drivers.fake import FakeGraspStore, FakeServo, FakeServoBus
from jetank.errors import ServoError, StorageError

JOINTS = {
    JointName.BASE: JointCalibration(
        servo_id=1, min_position=164, max_position=940, home_position=575
    ),
    JointName.CAMERA_TILT: JointCalibration(
        servo_id=5, min_position=244, max_position=783, home_position=535
    ),
}
GRIPPER = GripperCalibration(servo_id=4, open_position=955, closed_position=740, grip_overshoot=130)
MOMENT = datetime(2026, 11, 22, 9, 15, 0, 42, tzinfo=timezone.utc)
LABELS = TeachLabels(
    animal="sheep",
    stage=TeachStage.GRASP,
    posture=ToyPosture.STANDING,
    grip_point=GripPoint.BODY,
    result=GraspResult.GOOD,
)


@pytest.fixture
def bus() -> FakeServoBus:
    return FakeServoBus(
        servos={1: FakeServo(position=600), 4: FakeServo(position=900), 5: FakeServo(position=650)}
    )


@pytest.fixture
def gripper(bus: FakeServoBus) -> GripperController:
    return GripperController(bus, GRIPPER, move_time_s=0.8)


@pytest.fixture
def store() -> FakeGraspStore:
    return FakeGraspStore()


def make_teacher(
    bus: FakeServoBus,
    gripper: GripperController | None,
    store: FakeGraspStore,
    image: bytes | None = b"jpeg",
) -> GraspTeacher:
    arm = ArmController(bus, JOINTS, speed_cps=150.0)
    return GraspTeacher(arm, gripper, store, lambda: image, lambda: MOMENT, holding_margin=80)


def test_take_saves_labels_joints_gripper_and_image(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    gripper.shut()
    bus.servos[4].position = 760  # the toy stops the jaws
    record = make_teacher(bus, gripper, store).take(LABELS)
    assert record.record_id == "20261122-091500-000042"
    assert record.recorded_at == MOMENT
    assert record.labels == LABELS
    assert {j: s.position for j, s in record.joints.items()} == {
        JointName.BASE: 600,
        JointName.CAMERA_TILT: 650,
    }
    assert record.gripper is not None
    assert record.gripper.servo.position == 760
    assert record.gripper.target == 610
    assert record.gripper.holding
    assert record.image_file == f"{record.record_id}.jpg"
    assert store.records == [record]
    assert store.images[record.record_id] == b"jpeg"


def test_taking_a_pose_moves_nothing(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    make_teacher(bus, gripper, store).take(LABELS)
    assert bus.moves == []
    assert bus.torque_log == []


def test_an_empty_closed_gripper_is_not_holding(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    gripper.shut()
    bus.servos[4].position = 655  # jaws pressed together
    record = make_teacher(bus, gripper, store).take(LABELS)
    assert record.gripper is not None
    assert not record.gripper.holding


def test_a_limp_gripper_is_recorded_without_target(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    record = make_teacher(bus, gripper, store).take(LABELS)
    assert record.gripper is not None
    assert record.gripper.target is None
    assert not record.gripper.holding


def test_without_gripper_or_image_the_record_has_neither(
    bus: FakeServoBus, store: FakeGraspStore
) -> None:
    record = make_teacher(bus, None, store, image=None).take(LABELS)
    assert record.gripper is None
    assert record.image_file is None


def test_a_servo_that_does_not_answer_saves_nothing(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    bus.fail_with = ServoError("servo 5 did not answer")
    with pytest.raises(ServoError):
        make_teacher(bus, gripper, store).take(LABELS)
    assert store.records == []


def test_storage_failure_is_raised(
    bus: FakeServoBus, gripper: GripperController, store: FakeGraspStore
) -> None:
    store.fail_with = StorageError("disk full")
    with pytest.raises(StorageError, match="disk full"):
        make_teacher(bus, gripper, store).take(LABELS)


def test_holding_margin_must_be_positive(bus: FakeServoBus, store: FakeGraspStore) -> None:
    arm = ArmController(bus, JOINTS, speed_cps=150.0)
    with pytest.raises(ValueError, match="holding margin"):
        GraspTeacher(arm, None, store, lambda: None, lambda: MOMENT, holding_margin=0)
