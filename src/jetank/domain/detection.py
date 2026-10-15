"""
Detection and tracking value objects, plus the policy for how each object class is treated.

Layer: domain

Box coordinates are normalised to the frame: (0, 0) is the top-left corner, (1, 1) the
bottom-right. This keeps everything above the detector independent of image resolution.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")


class Response(str, Enum):
    """What the robot does about an object class."""

    CHASE = "chase"  # drive toward it to scare it away
    ALERT = "alert"  # record and notify, but do not move
    IGNORE = "ignore"


class BoundingBox(BaseModel):
    """Axis-aligned box in normalised frame coordinates."""

    model_config = _VALUE_OBJECT

    x1: Unit
    y1: Unit
    x2: Unit
    y2: Unit

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box corners must satisfy x1 <= x2 and y1 <= y2")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Fraction of the frame covered by the box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def iou(self, other: BoundingBox) -> float:
        """Intersection over union with ``other`` (0 when disjoint or both empty)."""
        inter_w = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        inter_h = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


class Detection(BaseModel):
    """One object found in one frame."""

    model_config = _VALUE_OBJECT

    label: str = Field(min_length=1)
    confidence: Unit
    box: BoundingBox


class Track(BaseModel):
    """An object followed across frames.

    Attributes:
        track_id: Unique, increasing identifier.
        hits: Number of frames in which the object was matched.
        first_seen_s / last_seen_s: Monotonic timestamps of the first and latest match.
        confirmed: True once ``hits`` reached the tracker's confirmation threshold.
    """

    model_config = _VALUE_OBJECT

    track_id: int = Field(ge=1)
    label: str
    box: BoundingBox
    confidence: Unit
    hits: int = Field(ge=1)
    first_seen_s: float
    last_seen_s: float
    confirmed: bool = False


class IntruderEvent(BaseModel):
    """A confirmed intruder, as recorded and reported."""

    model_config = _VALUE_OBJECT

    track_id: int
    label: str
    response: Response
    confidence: Unit
    box: BoundingBox
    timestamp_s: float
