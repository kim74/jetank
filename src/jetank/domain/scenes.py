"""
Scene recording: camera images of the floors, collected to train the toy and floor detector.

Layer: domain

The operator drives the robot around with the arm in the *search* pose and records scenes,
one at a time or automatically every few seconds. Each :class:`SceneRecord` holds the
operator's labels (room, light, whether toy animals are in view) and the arm pose the image
was taken from; the image is saved next to it. Wrong labels are fixed afterwards on the scene
review page (``jetank scenes review``). Boxes and floor outlines are drawn later, in a
labelling tool, not here.

Client → robot: :class:`SceneMessage`. Robot → client: :class:`SceneReport` inside the teleop
status.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jetank.domain.records import ImageRecord, LabelName, RecordId
from jetank.domain.servo import ArmPose

if TYPE_CHECKING:
    from collections.abc import Iterable

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")

#: Longest note an operator can attach to a scene.
MAX_SCENE_NOTE_LENGTH: Final = 200


class Lighting(str, Enum):
    """How the scene is lit, so the training set can be checked for variety."""

    DAYLIGHT = "daylight"
    LAMP = "lamp"
    DIM = "dim"


class SceneLabels(BaseModel):
    """What the operator says about the scenes being recorded.

    Attributes:
        room: Where, e.g. "kitchen" or "living room".
        lighting: How the scene is lit.
        has_animals: Whether toy animals are in view; ``False`` marks a "nothing here" scene
            (bare floor, other toys, socks, cables…) that teaches the detector what is not a
            toy animal. Other toys may also lie next to the animals in a scene with animals.
        note: Free text, at most ``MAX_SCENE_NOTE_LENGTH`` characters.
    """

    model_config = _VALUE_OBJECT

    room: LabelName
    lighting: Lighting
    has_animals: bool = True
    note: str = Field(default="", max_length=MAX_SCENE_NOTE_LENGTH)


class SceneRecord(ImageRecord):
    """One recorded scene: the operator's labels and the pose the camera looked from.

    Attributes (besides those of :class:`~jetank.domain.records.ImageRecord`):
        labels: What the operator said about it.
        arm_pose: Measured position of every calibrated arm joint (the camera tilt above all),
            or ``None`` when the robot has no arm or it could not be read.
    """

    labels: SceneLabels
    arm_pose: ArmPose | None = None

    def relabeled(self, labels: SceneLabels) -> SceneRecord:
        """The same scene with corrected labels."""
        return self.model_copy(update={"labels": labels})


class SceneAction(str, Enum):
    """What a :class:`SceneMessage` asks for."""

    TAKE = "take"  # record one scene now
    AUTO_START = "auto_start"  # record one every few seconds, until stopped or a time limit
    AUTO_STOP = "auto_stop"


class SceneMessage(BaseModel):
    """Request to record scenes. Accepted from the driver only; nothing moves.

    Attributes:
        action: What to do.
        labels: Labels for the scenes; required with ``take`` and ``auto_start`` and not
            allowed with ``auto_stop``.
    """

    model_config = _VALUE_OBJECT

    type: Literal["scene"]
    action: SceneAction
    labels: SceneLabels | None = None

    @model_validator(mode="after")
    def _labels_unless_stopping(self) -> SceneMessage:
        if (self.action is SceneAction.AUTO_STOP) == (self.labels is not None):
            raise ValueError("labels are required to record scenes and not allowed to stop")
        return self


class SceneReport(BaseModel):
    """Scene recording as shown to the operator.

    Attributes:
        auto: Whether scenes are being recorded automatically.
        auto_remaining_s: Seconds until automatic recording stops by itself, while it runs.
        taken: Scenes recorded since the teleop server started.
        last_record_id: ID of the newest scene recorded, if any.
        error: Why the last recording failed (automatic recording then stops), if it did.
    """

    model_config = _VALUE_OBJECT

    auto: bool = False
    auto_remaining_s: float | None = Field(default=None, ge=0.0)
    taken: int = Field(default=0, ge=0)
    last_record_id: RecordId | None = None
    error: str | None = None


class SceneSummary(BaseModel):
    """How varied the recorded scenes are.

    Attributes:
        total: Number of scenes.
        rooms: Scenes per room.
        lighting: Scenes per kind of light.
        with_animals: Scenes with animals in view (the rest are "nothing here" scenes).
    """

    model_config = _VALUE_OBJECT

    total: int
    rooms: dict[str, int]
    lighting: dict[Lighting, int]
    with_animals: int


def summarize_scenes(records: Iterable[SceneRecord]) -> SceneSummary:
    """Count the scenes per room and kind of light, and those with animals in view."""
    labels = [record.labels for record in records]
    return SceneSummary(
        total=len(labels),
        rooms=dict(Counter(label.room for label in labels).most_common()),
        lighting=dict(Counter(label.lighting for label in labels).most_common()),
        with_animals=sum(label.has_animals for label in labels),
    )
