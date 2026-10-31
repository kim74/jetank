"""
Messages exchanged with a remote-control (teleop) client over the network.

Layer: domain

Client → robot: :class:`DriveMessage` (sent repeatedly while driving, so it doubles as the
heartbeat that keeps the drive watchdog fed), :class:`StopMessage` and
:class:`GripperMessage`.
Robot → client: :class:`TeleopStatus` after every message and on connect.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from jetank.domain.models import DriveCommand, TrackSpeeds, UnitSpeed
from jetank.domain.servo import Opening
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


class GripperAction(str, Enum):
    """What a :class:`GripperMessage` asks the gripper to do."""

    OPEN = "open"
    CLOSE = "close"
    SET = "set"
    RELEASE = "release"  # torque off: the gripper goes limp and lets go


class GripperMessage(BaseModel):
    """Request to move or release the gripper. Accepted from the driver only.

    Attributes:
        action: What to do.
        opening: Target for ``set`` (0.0 = closed, 1.0 = open); only allowed with ``set``.
    """

    model_config = _MESSAGE

    type: Literal["gripper"]
    action: GripperAction
    opening: Opening | None = None

    @model_validator(mode="after")
    def _opening_only_with_set(self) -> GripperMessage:
        if (self.action is GripperAction.SET) != (self.opening is not None):
            raise ValueError("an opening is required with 'set' and not allowed otherwise")
        return self


AnyClientMessage = DriveMessage | StopMessage | GripperMessage
ClientMessage = Annotated[
    Union[DriveMessage, StopMessage, GripperMessage],  # noqa: UP007
    Field(discriminator="type"),
]

_CLIENT_MESSAGE: Final[TypeAdapter[AnyClientMessage]] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes) -> AnyClientMessage:
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


class GripperReport(BaseModel):
    """Gripper state as last commanded (not read back, to keep status replies fast).

    Attributes:
        engaged: Whether the gripper is powered (holding a position) rather than limp.
        opening: Last commanded opening while engaged, else ``None``.
    """

    model_config = _MESSAGE

    engaged: bool
    opening: Opening | None = None


class TeleopStatus(BaseModel):
    """Robot state reported to a client.

    Attributes:
        role: The receiving client's role.
        moving: Whether the tracks are currently commanded to move.
        speeds: Track speeds last sent to the motors (after the speed limit).
        gripper: Gripper state, or ``None`` when the robot has no working gripper.
        error: Why the last message was rejected, if it was.
    """

    model_config = _MESSAGE

    type: Literal["status"] = "status"
    role: Role
    moving: bool
    speeds: TrackSpeeds
    gripper: GripperReport | None = None
    error: str | None = None
