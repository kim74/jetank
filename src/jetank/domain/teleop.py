"""
Messages exchanged with a remote-control (teleop) client over the network.

Layer: domain

Client → robot: :class:`DriveMessage` (sent repeatedly while driving, so it doubles as the
heartbeat that keeps the drive watchdog fed) and :class:`StopMessage`.
Robot → client: :class:`TeleopStatus` after every message and on connect.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from jetank.domain.models import DriveCommand, TrackSpeeds, UnitSpeed
from jetank.errors import ProtocolError

_MESSAGE: Final = ConfigDict(frozen=True, extra="forbid")


class DriveMessage(BaseModel):
    """Request to drive; see :class:`~jetank.domain.models.DriveCommand` for the axes."""

    model_config = _MESSAGE

    type: Literal["drive"]
    linear: UnitSpeed
    angular: UnitSpeed

    def to_command(self) -> DriveCommand:
        return DriveCommand(linear=self.linear, angular=self.angular)


class StopMessage(BaseModel):
    """Request to stop both tracks. Accepted from every client, not only the driver."""

    model_config = _MESSAGE

    type: Literal["stop"]


ClientMessage = Annotated[Union[DriveMessage, StopMessage], Field(discriminator="type")]  # noqa: UP007

_CLIENT_MESSAGE: Final[TypeAdapter[DriveMessage | StopMessage]] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> DriveMessage | StopMessage:
    """Decode one JSON message from a client.

    Raises:
        ProtocolError: The message is not valid JSON or does not match a known message.
    """
    try:
        return _CLIENT_MESSAGE.validate_json(raw)
    except ValidationError as exc:
        raise ProtocolError(f"invalid client message: {exc}") from exc


class Role(str, Enum):
    """What a connected client may do. Only one client drives at a time."""

    DRIVER = "driver"
    OBSERVER = "observer"


class TeleopStatus(BaseModel):
    """Robot state reported to a client.

    Attributes:
        role: The receiving client's role.
        moving: Whether the tracks are currently commanded to move.
        speeds: Track speeds last sent to the motors (after the speed limit).
        error: Why the last message was rejected, if it was.
    """

    model_config = _MESSAGE

    type: Literal["status"] = "status"
    role: Role
    moving: bool
    speeds: TrackSpeeds
    error: str | None = None
