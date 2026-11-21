"""
Messages exchanged with a remote-control (teleop) client over the network.

Layer: domain

Client → robot: :class:`DriveMessage` (sent repeatedly while driving, so it doubles as the
heartbeat that keeps the drive watchdog fed), :class:`StopMessage`, :class:`GripperMessage`,
:class:`ArmJogMessage` (sent repeatedly while an arm key is held), :class:`SleepMessage` and
:class:`TeachMessage`.
Robot → client: :class:`TeleopStatus` after every message and on connect.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from jetank.domain.models import DriveCommand, TrackSpeeds, UnitSpeed
from jetank.domain.servo import JointName, Opening, ServoPosition
from jetank.domain.teaching import GraspRecord, GraspResult, RecordId, TeachLabels, TeachStage
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


class ArmJogMessage(BaseModel):
    """Request to move one arm joint a small step. Accepted from the driver only.

    Sent repeatedly while the operator holds a key; the joint stops soon after the
    messages stop.

    Attributes:
        joint: Joint to move.
        direction: Sign of the change in raw servo counts. Which way that turns the joint
            depends on how its servo is mounted (on the JETANK: base left, shoulder up and
            elbow folding are all -1).
    """

    model_config = _MESSAGE

    type: Literal["arm"]
    joint: JointName
    direction: Literal[-1, 1]


class SleepMessage(BaseModel):
    """Request to put the robot to sleep. Accepted from the driver only.

    The tracks stop, the gripper goes limp, and the arm moves home and then goes limp. The
    next arm or gripper command wakes the robot up again.
    """

    model_config = _MESSAGE

    type: Literal["sleep"]


class TeachMessage(BaseModel):
    """Request to record the arm's current pose with the operator's labels. Driver only.

    Nothing moves: the robot reads every joint and the gripper, and saves them with the
    labels and the current camera image.
    """

    model_config = _MESSAGE

    type: Literal["teach"]
    labels: TeachLabels


AnyClientMessage = (
    DriveMessage | StopMessage | GripperMessage | ArmJogMessage | SleepMessage | TeachMessage
)
ClientMessage = Annotated[
    Union[  # noqa: UP007
        DriveMessage, StopMessage, GripperMessage, ArmJogMessage, SleepMessage, TeachMessage
    ],
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


class ArmReport(BaseModel):
    """Arm state as last commanded (not read back).

    Attributes:
        targets: Last commanded position of every powered joint, in raw counts; joints
            not listed are limp.
    """

    model_config = _MESSAGE

    targets: dict[JointName, ServoPosition] = Field(default_factory=dict)


class TeachReport(BaseModel):
    """Summary of the last pose recorded, shown to the operator as a confirmation.

    Attributes:
        record_id: ID of the saved record.
        animal / stage / result: The labels it was saved with.
        gripper_position: Gripper position measured, or ``None`` without a gripper.
        holding: Whether the gripper was holding something.
        image_saved: Whether a camera image was saved with it.
    """

    model_config = _MESSAGE

    record_id: RecordId
    animal: str
    stage: TeachStage
    result: GraspResult
    gripper_position: ServoPosition | None = None
    holding: bool = False
    image_saved: bool = False

    @classmethod
    def from_record(cls, record: GraspRecord) -> TeachReport:
        gripper = record.gripper
        return cls(
            record_id=record.record_id,
            animal=record.labels.animal,
            stage=record.labels.stage,
            result=record.labels.result,
            gripper_position=None if gripper is None else gripper.servo.position,
            holding=gripper is not None and gripper.holding,
            image_saved=record.image_file is not None,
        )


class TeleopStatus(BaseModel):
    """Robot state reported to a client.

    Attributes:
        role: The receiving client's role.
        moving: Whether the tracks are currently commanded to move.
        speeds: Track speeds last sent to the motors (after the speed limit).
        gripper: Gripper state, or ``None`` when the robot has no working gripper.
        arm: Arm state, or ``None`` when the robot has no working arm.
        teach_enabled: Whether poses can be recorded (:class:`TeachMessage`).
        last_teach: The last pose recorded in this session, if any.
        asleep: Whether the robot was put to sleep (by request or after being idle) and no
            arm or gripper command has woken it since. The arm may still be on its way home.
        error: Why the last message was rejected, if it was.
    """

    model_config = _MESSAGE

    type: Literal["status"] = "status"
    role: Role
    moving: bool
    speeds: TrackSpeeds
    gripper: GripperReport | None = None
    arm: ArmReport | None = None
    teach_enabled: bool = False
    last_teach: TeachReport | None = None
    asleep: bool = False
    error: str | None = None
