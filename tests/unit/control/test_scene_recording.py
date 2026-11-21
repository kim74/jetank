from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jetank.control.arm import ArmController
from jetank.control.scene_recording import SceneRecorder
from jetank.domain.scenes import Lighting, SceneAction, SceneLabels, SceneMessage
from jetank.domain.servo import ArmPose, JointCalibration, JointName
from jetank.drivers.fake import FakeClock, FakeSceneStore, FakeServo, FakeServoBus
from jetank.errors import CameraError, ServoError, StorageError

PERIOD_S = 2.0
LIMIT_S = 60.0
LABELS = SceneLabels(room="kitchen", lighting=Lighting.DAYLIGHT)
TILT = {
    JointName.CAMERA_TILT: JointCalibration(
        servo_id=5, min_position=244, max_position=783, home_position=535
    )
}


def message(action: SceneAction) -> SceneMessage:
    labels = None if action is SceneAction.AUTO_STOP else LABELS
    return SceneMessage(type="scene", action=action, labels=labels)


TAKE = message(SceneAction.TAKE)
START = message(SceneAction.AUTO_START)
STOP = message(SceneAction.AUTO_STOP)


class Camera:
    """Snapshot source whose image can be taken away or made to fail."""

    def __init__(self) -> None:
        self.jpeg: bytes | None = b"jpeg"
        self.fail_with: CameraError | None = None

    def __call__(self) -> bytes | None:
        if self.fail_with is not None:
            raise self.fail_with
        return self.jpeg


class WallClock:
    """Wall-clock time that moves on by one second per call, so record IDs differ."""

    def __init__(self) -> None:
        self._moment = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self._moment += timedelta(seconds=1)
        return self._moment


@pytest.fixture
def store() -> FakeSceneStore:
    return FakeSceneStore()


@pytest.fixture
def camera() -> Camera:
    return Camera()


@pytest.fixture
def bus() -> FakeServoBus:
    return FakeServoBus(servos={5: FakeServo(position=572)})


@pytest.fixture
def recorder(
    store: FakeSceneStore, camera: Camera, bus: FakeServoBus, clock: FakeClock
) -> SceneRecorder:
    arm = ArmController(bus, TILT, speed_cps=150.0)
    return SceneRecorder(
        store,
        camera,
        arm,
        now=WallClock(),
        clock=clock,
        auto_period_s=PERIOD_S,
        auto_limit_s=LIMIT_S,
    )


def test_take_saves_the_image_labels_and_camera_pose(
    recorder: SceneRecorder, store: FakeSceneStore
) -> None:
    assert recorder.handle(TAKE) is None
    [record] = store.records
    assert record.labels == LABELS
    assert record.arm_pose == ArmPose(positions={JointName.CAMERA_TILT: 572})
    assert store.images[record.record_id] == b"jpeg"
    report = recorder.report()
    assert report.taken == 1
    assert report.last_record_id == record.record_id
    assert not report.auto


def test_taking_moves_nothing(recorder: SceneRecorder, bus: FakeServoBus) -> None:
    recorder.handle(TAKE)
    assert bus.moves == []
    assert bus.torque_log == []


def test_without_an_image_nothing_is_saved(
    recorder: SceneRecorder, store: FakeSceneStore, camera: Camera
) -> None:
    camera.jpeg = None
    error = recorder.handle(TAKE)
    assert error is not None
    assert "no camera image" in error
    assert recorder.report().error == error
    assert store.records == []


def test_a_silent_arm_still_lets_the_scene_be_saved(
    recorder: SceneRecorder, store: FakeSceneStore, bus: FakeServoBus
) -> None:
    bus.fail_with = ServoError("servo 5 does not answer")
    assert recorder.handle(TAKE) is None
    assert store.records[0].arm_pose is None


def test_without_an_arm_scenes_have_no_pose(
    store: FakeSceneStore, camera: Camera, clock: FakeClock
) -> None:
    recorder = SceneRecorder(
        store, camera, None, now=WallClock(), clock=clock, auto_period_s=1, auto_limit_s=1
    )
    recorder.take(LABELS)
    assert store.records[0].arm_pose is None


def test_auto_records_at_once_and_then_every_period(
    recorder: SceneRecorder, store: FakeSceneStore, clock: FakeClock
) -> None:
    assert recorder.handle(START) is None
    assert recorder.auto
    assert len(store.records) == 1
    clock.advance(PERIOD_S - 0.1)
    recorder.tick()
    assert len(store.records) == 1
    clock.advance(0.2)
    recorder.tick()
    assert len(store.records) == 2
    report = recorder.report()
    assert report.auto_remaining_s == pytest.approx(LIMIT_S - PERIOD_S - 0.1)


def test_auto_stops_on_request(
    recorder: SceneRecorder, store: FakeSceneStore, clock: FakeClock
) -> None:
    recorder.handle(START)
    assert recorder.handle(STOP) is None
    clock.advance(PERIOD_S * 3)
    recorder.tick()
    assert len(store.records) == 1
    assert recorder.report().auto_remaining_s is None


def test_auto_stops_by_itself_at_the_time_limit(
    recorder: SceneRecorder, store: FakeSceneStore, clock: FakeClock
) -> None:
    recorder.handle(START)
    for _ in range(int(LIMIT_S / PERIOD_S) + 10):
        clock.advance(PERIOD_S)
        recorder.tick()
    assert not recorder.auto
    assert len(store.records) == LIMIT_S / PERIOD_S  # the one at the limit is not taken


def test_a_failed_auto_recording_stops_auto_and_is_reported(
    recorder: SceneRecorder, store: FakeSceneStore, clock: FakeClock
) -> None:
    recorder.handle(START)
    store.fail_with = StorageError("disk full")
    clock.advance(PERIOD_S)
    recorder.tick()
    assert not recorder.auto
    error = recorder.report().error
    assert error is not None
    assert "disk full" in error


def test_auto_does_not_start_when_the_first_scene_fails(
    recorder: SceneRecorder, camera: Camera
) -> None:
    camera.fail_with = CameraError("could not encode the frame as JPEG")
    assert recorder.handle(START) is not None
    assert not recorder.auto


def test_a_good_recording_clears_the_last_error(recorder: SceneRecorder, camera: Camera) -> None:
    camera.jpeg = None
    recorder.handle(TAKE)
    camera.jpeg = b"jpeg"
    recorder.handle(TAKE)
    assert recorder.report().error is None


def test_tick_without_auto_does_nothing(recorder: SceneRecorder, store: FakeSceneStore) -> None:
    recorder.tick()
    recorder.stop_auto()
    assert store.records == []


@pytest.mark.parametrize(("period_s", "limit_s"), [(0.0, 60.0), (2.0, 0.0)])
def test_period_and_limit_must_be_positive(
    store: FakeSceneStore, camera: Camera, clock: FakeClock, period_s: float, limit_s: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        SceneRecorder(
            store,
            camera,
            None,
            now=WallClock(),
            clock=clock,
            auto_period_s=period_s,
            auto_limit_s=limit_s,
        )
