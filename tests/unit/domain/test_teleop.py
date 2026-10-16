from __future__ import annotations

import pytest

from jetank.domain.models import DriveCommand, TrackSpeeds
from jetank.domain.teleop import (
    DriveMessage,
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
        "error": None,
    }
