from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jetank.domain.records import record_id_for
from jetank.domain.scenes import (
    Lighting,
    SceneAction,
    SceneLabels,
    SceneMessage,
    SceneRecord,
    SceneReport,
    summarize_scenes,
)
from jetank.domain.servo import ArmPose, JointName
from jetank.domain.teleop import PingMessage, PoseMessage, parse_client_message
from jetank.errors import ProtocolError

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)
KITCHEN = SceneLabels(room="kitchen", lighting=Lighting.DAYLIGHT)


def scene(labels: SceneLabels, micro: int = 0) -> SceneRecord:
    moment = T0.replace(microsecond=micro)
    return SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=labels)


def test_room_names_are_normalised_and_toys_assumed_in_view() -> None:
    labels = SceneLabels.model_validate({"room": "  Living Room ", "lighting": "lamp"})
    assert labels.room == "living room"
    assert labels.has_animals


@pytest.mark.parametrize("room", ["", "../etc", "a" * 41])
def test_unusable_room_names_are_rejected(room: str) -> None:
    with pytest.raises(ValidationError):
        SceneLabels(room=room, lighting=Lighting.DIM)


def test_record_round_trips_through_json_with_its_pose() -> None:
    record = scene(KITCHEN).model_copy(
        update={"arm_pose": ArmPose(positions={JointName.CAMERA_TILT: 572})}
    )
    assert SceneRecord.model_validate_json(record.model_dump_json()) == record


def test_relabeled_keeps_everything_but_the_labels() -> None:
    record = scene(KITCHEN).model_copy(
        update={"arm_pose": ArmPose(positions={JointName.CAMERA_TILT: 572})}
    )
    hall = SceneLabels(room="hall", lighting=Lighting.DIM, has_animals=False)
    relabeled = record.relabeled(hall)
    assert relabeled.labels == hall
    assert relabeled.model_dump(exclude={"labels"}) == record.model_dump(exclude={"labels"})


def test_parses_scene_messages() -> None:
    take = parse_client_message(
        '{"type": "scene", "action": "take", '
        '"labels": {"room": "kitchen", "lighting": "dim", "has_animals": false}}'
    )
    assert isinstance(take, SceneMessage)
    assert take.action is SceneAction.TAKE
    assert take.labels == SceneLabels(room="kitchen", lighting=Lighting.DIM, has_animals=False)
    stop = parse_client_message('{"type": "scene", "action": "auto_stop"}')
    assert isinstance(stop, SceneMessage)
    assert stop.labels is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"type": "scene", "action": "take"}',
        '{"type": "scene", "action": "auto_start"}',
        '{"type": "scene", "action": "auto_stop", "labels": {"room": "x", "lighting": "dim"}}',
        '{"type": "scene", "action": "take", "labels": {"room": "x", "lighting": "neon"}}',
    ],
)
def test_rejects_invalid_scene_messages(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_client_message(raw)


def test_parses_pose_and_ping_messages() -> None:
    assert parse_client_message('{"type": "pose", "name": "search"}') == PoseMessage(
        type="pose", name="search"
    )
    assert parse_client_message('{"type": "ping"}') == PingMessage(type="ping")


@pytest.mark.parametrize("name", ["", "Search", "../x", "a" * 41])
def test_rejects_unusable_pose_names(name: str) -> None:
    with pytest.raises(ProtocolError):
        parse_client_message(f'{{"type": "pose", "name": "{name}"}}')


def test_report_defaults_to_idle() -> None:
    report = SceneReport()
    assert not report.auto
    assert report.taken == 0


def test_summary_counts_rooms_lighting_and_scenes_with_animals() -> None:
    hall_dim = SceneLabels(room="hall", lighting=Lighting.DIM, has_animals=False)
    records = [scene(KITCHEN, 1), scene(KITCHEN, 2), scene(hall_dim, 3)]
    summary = summarize_scenes(records)
    assert summary.total == 3
    assert summary.rooms == {"kitchen": 2, "hall": 1}
    assert summary.lighting == {Lighting.DAYLIGHT: 2, Lighting.DIM: 1}
    assert summary.with_animals == 2


def test_summary_of_nothing_is_empty() -> None:
    summary = summarize_scenes([])
    assert summary.total == 0
    assert summary.rooms == {}
