from __future__ import annotations

import pytest

from jetank.domain.models import DriveCommand, TrackSpeeds
from jetank.domain.servo import JointName
from jetank.domain.teleop import (
    ArmJogMessage,
    ArmReport,
    DriveMessage,
    GripperAction,
    GripperMessage,
    GripperReport,
    Role,
    StopMessage,
    TeleopStatus,
    parse_client_message,
)
from jetank.errors import ProtocolError


def test_parses_drive_message() -> None:
    message = parse_client_message('{"type": "drive", "linear": 0.5, "angular": -0.25}')
    assert message == DriveMessage(type="drive", linear=0.5, angular=-0.25)
    assert isinstance(message, DriveMessage)
    assert message.to_command() == DriveCommand(linear=0.5, angular=-0.25)


def test_parses_stop_message_from_bytes() -> None:
    assert parse_client_message(b'{"type": "stop"}') == StopMessage(type="stop")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"linear": 0.5, "angular": 0.0}',
        '{"type": "jump"}',
        '{"type": "drive", "linear": 1.5, "angular": 0.0}',
        '{"type": "drive", "linear": 0.5}',
        '{"type": "drive", "linear": NaN, "angular": 0.0}',
        '{"type": "stop", "force": true}',
        '{"type": "arm", "joint": "elbow", "direction": 2}',
        '{"type": "arm", "joint": "elbow", "direction": 0}',
        '{"type": "arm", "joint": "wrist", "direction": 1}',
        '{"type": "arm", "joint": "elbow"}',
    ],
)
def test_rejects_invalid_messages(raw: str) -> None:
    with pytest.raises(ProtocolError, match="invalid client message"):
        parse_client_message(raw)


def test_status_serialises_to_json() -> None:
    status = TeleopStatus(role=Role.DRIVER, moving=True, speeds=TrackSpeeds(left=0.25, right=0.5))
    assert status.model_dump(mode="json") == {
        "type": "status",
        "role": "driver",
        "moving": True,
        "speeds": {"left": 0.25, "right": 0.5},
        "gripper": None,
        "arm": None,
        "error": None,
    }


@pytest.mark.parametrize(
    ("raw", "action", "opening"),
    [
        ('{"type": "gripper", "action": "open"}', GripperAction.OPEN, None),
        ('{"type": "gripper", "action": "release"}', GripperAction.RELEASE, None),
        ('{"type": "gripper", "action": "set", "opening": 0.25}', GripperAction.SET, 0.25),
    ],
)
def test_parses_gripper_messages(raw: str, action: GripperAction, opening: float | None) -> None:
    message = parse_client_message(raw)
    assert isinstance(message, GripperMessage)
    assert (message.action, message.opening) == (action, opening)


@pytest.mark.parametrize(
    "raw",
    [
        '{"type": "gripper", "action": "set"}',
        '{"type": "gripper", "action": "open", "opening": 0.5}',
        '{"type": "gripper", "action": "set", "opening": 1.5}',
        '{"type": "gripper", "action": "squeeze"}',
    ],
)
def test_rejects_invalid_gripper_messages(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_client_message(raw)


def test_status_reports_gripper() -> None:
    status = TeleopStatus(
        role=Role.DRIVER,
        moving=False,
        speeds=TrackSpeeds.stopped(),
        gripper=GripperReport(engaged=True, opening=0.5),
    )
    assert status.model_dump(mode="json")["gripper"] == {"engaged": True, "opening": 0.5}


def test_parses_arm_jog_message() -> None:
    message = parse_client_message('{"type": "arm", "joint": "shoulder", "direction": -1}')
    assert isinstance(message, ArmJogMessage)
    assert message.joint is JointName.SHOULDER
    assert message.direction == -1


def test_status_reports_arm_targets_by_joint_name() -> None:
    status = TeleopStatus(
        role=Role.DRIVER,
        moving=False,
        speeds=TrackSpeeds.stopped(),
        arm=ArmReport(targets={JointName.BASE: 575}),
    )
    assert '"arm":{"targets":{"base":575}}' in status.model_dump_json()
