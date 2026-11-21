"""TeleopSession: named arm poses, ping and scene recording."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from jetank.control.arm import ArmController
from jetank.control.drive import DriveController
from jetank.control.scene_recording import SceneRecorder
from jetank.control.teleop import TeleopSession
from jetank.domain.scenes import Lighting, SceneAction, SceneLabels, SceneMessage, SceneReport
from jetank.domain.servo import ArmPose, JointCalibration, JointName
from jetank.domain.teleop import (
    DriveMessage,
    PingMessage,
    PoseMessage,
    Role,
    SleepMessage,
    StopMessage,
)
from jetank.drivers.fake import FakeClock, FakeMotorDriver, FakeSceneStore, FakeServo, FakeServoBus
from jetank.errors import ServoError

if TYPE_CHECKING:
    from jetank.config.settings import DriveConfig

IDLE_SLEEP_S = 120.0
PERIOD_S = 2.0
LIMIT_S = 600.0
JOINTS = {
    JointName.SHOULDER: JointCalibration(
        servo_id=2, min_position=223, max_position=948, home_position=870
    ),
    JointName.CAMERA_TILT: JointCalibration(
        servo_id=5, min_position=244, max_position=783, home_position=535
    ),
}
SEARCH = ArmPose(positions={JointName.SHOULDER: 724, JointName.CAMERA_TILT: 572})
LABELS = SceneLabels(room="kitchen", lighting=Lighting.LAMP)
TAKE = SceneMessage(type="scene", action=SceneAction.TAKE, labels=LABELS)
AUTO = SceneMessage(type="scene", action=SceneAction.AUTO_START, labels=LABELS)
AUTO_STOP = SceneMessage(type="scene", action=SceneAction.AUTO_STOP)
TO_SEARCH = PoseMessage(type="pose", name="search")


T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def bus() -> FakeServoBus:
    return FakeServoBus(servos={2: FakeServo(position=870), 5: FakeServo(position=535)})


@pytest.fixture
def store() -> FakeSceneStore:
    return FakeSceneStore()


@pytest.fixture
def session(
    motors: FakeMotorDriver,
    clock: FakeClock,
    drive_config: DriveConfig,
    bus: FakeServoBus,
    store: FakeSceneStore,
) -> TeleopSession:
    arm = ArmController(bus, JOINTS, speed_cps=150.0)
    moments = (T0 + timedelta(seconds=i) for i in itertools.count())
    scenes = SceneRecorder(
        store,
        lambda: b"jpeg",
        arm,
        now=lambda: next(moments),
        clock=clock,
        auto_period_s=PERIOD_S,
        auto_limit_s=LIMIT_S,
    )
    return TeleopSession(
        DriveController(motors, clock, drive_config),
        arm=arm,
        clock=clock,
        idle_sleep_s=IDLE_SLEEP_S,
        poses={"search": SEARCH},
        scenes=scenes,
    )


@pytest.fixture
def plain_session(
    motors: FakeMotorDriver, clock: FakeClock, drive_config: DriveConfig
) -> TeleopSession:
    return TeleopSession(DriveController(motors, clock, drive_config), clock=clock)


# --- poses ---------------------------------------------------------------------------------


def test_driver_moves_the_arm_to_a_named_pose(session: TeleopSession, bus: FakeServoBus) -> None:
    session.connect("a")
    status = session.handle("a", TO_SEARCH)
    assert status.error is None
    assert bus.servos[2].position == 724
    assert bus.servos[5].position == 572
    assert status.arm is not None
    assert status.arm.targets == {JointName.SHOULDER: 724, JointName.CAMERA_TILT: 572}
    assert status.poses == ["search"]


def test_observer_cannot_move_the_arm_to_a_pose(session: TeleopSession, bus: FakeServoBus) -> None:
    session.connect("a")
    session.connect("b")
    assert session.handle("b", TO_SEARCH).error == "another client is driving"
    assert bus.moves == []


def test_unknown_pose_is_reported(session: TeleopSession, bus: FakeServoBus) -> None:
    session.connect("a")
    error = session.handle("a", PoseMessage(type="pose", name="dance")).error
    assert error is not None
    assert "dance" in error
    assert bus.moves == []


def test_pose_failure_is_reported_and_driving_continues(
    session: TeleopSession, bus: FakeServoBus
) -> None:
    session.connect("a")
    bus.fail_with = ServoError("servo 2 does not answer")
    error = session.handle("a", TO_SEARCH).error
    assert error is not None
    assert error.startswith("arm failed")
    assert session.handle("a", DriveMessage(type="drive", linear=1.0, angular=0.0)).moving


def test_without_an_arm_there_are_no_poses(plain_session: TeleopSession) -> None:
    plain_session.connect("a")
    status = plain_session.handle("a", TO_SEARCH)
    assert status.error == "this robot has no arm"
    assert status.poses == []


def test_a_pose_wakes_the_robot(session: TeleopSession) -> None:
    session.connect("a")
    session.handle("a", SleepMessage(type="sleep"))
    assert not session.handle("a", TO_SEARCH).asleep


# --- ping ----------------------------------------------------------------------------------


def test_any_client_can_ping_for_a_status(session: TeleopSession) -> None:
    session.connect("a")
    session.connect("b")
    status = session.handle("b", PingMessage(type="ping"))
    assert status.role is Role.OBSERVER
    assert status.error is None


# --- scenes --------------------------------------------------------------------------------


def test_driver_records_a_scene(session: TeleopSession, store: FakeSceneStore) -> None:
    session.connect("a")
    status = session.handle("a", TAKE)
    assert status.error is None
    assert status.scenes is not None
    assert status.scenes.taken == 1
    assert store.records[0].labels == LABELS


def test_observer_cannot_record_scenes(session: TeleopSession, store: FakeSceneStore) -> None:
    session.connect("a")
    session.connect("b")
    assert session.handle("b", TAKE).error == "another client is driving"
    assert store.records == []


def test_without_a_recorder_scenes_are_refused(plain_session: TeleopSession) -> None:
    plain_session.connect("a")
    status = plain_session.handle("a", TAKE)
    assert status.error == "scene recording is not available"
    assert status.scenes is None


def test_auto_recording_runs_on_tick(
    session: TeleopSession, store: FakeSceneStore, clock: FakeClock
) -> None:
    session.connect("a")
    session.handle("a", AUTO)
    clock.advance(PERIOD_S)
    session.tick()
    assert len(store.records) == 2
    status = session.handle("a", AUTO_STOP)
    assert status.scenes == SceneReport(
        taken=2, last_record_id=store.records[-1].record_id, auto=False
    )


def test_auto_recording_keeps_the_arm_awake(
    session: TeleopSession, bus: FakeServoBus, clock: FakeClock
) -> None:
    session.connect("a")
    session.handle("a", TO_SEARCH)
    session.handle("a", AUTO)
    for _ in range(int(IDLE_SLEEP_S / PERIOD_S) * 2):
        clock.advance(PERIOD_S)
        session.tick()
    assert not session.status("a").asleep
    assert bus.servos[2].position == 724  # the camera still looks from the search pose


def test_the_arm_falls_asleep_once_auto_recording_ends(
    session: TeleopSession, bus: FakeServoBus, clock: FakeClock
) -> None:
    session.connect("a")
    session.handle("a", TO_SEARCH)
    session.handle("a", AUTO)
    session.handle("a", AUTO_STOP)
    clock.advance(IDLE_SLEEP_S)
    session.tick()
    assert session.status("a").asleep


def test_stop_does_not_end_auto_recording(session: TeleopSession) -> None:
    session.connect("a")
    session.handle("a", AUTO)
    status = session.handle("a", StopMessage(type="stop"))
    assert status.scenes is not None
    assert status.scenes.auto


def test_sleep_ends_auto_recording(session: TeleopSession) -> None:
    session.connect("a")
    session.handle("a", AUTO)
    status = session.handle("a", SleepMessage(type="sleep"))
    assert status.scenes is not None
    assert not status.scenes.auto


def test_driver_leaving_ends_auto_recording(
    session: TeleopSession, store: FakeSceneStore, clock: FakeClock
) -> None:
    session.connect("a")
    session.handle("a", AUTO)
    session.disconnect("a")
    clock.advance(PERIOD_S * 5)
    session.tick()
    assert len(store.records) == 1
