"""
Value objects shared across layers: drive commands, track speeds and camera frames.

Layer: domain
"""

from __future__ import annotations

from typing import Annotated, Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Normalised actuator command: -1.0 = full reverse, 0.0 = stop, 1.0 = full forward.
UnitSpeed = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")


class TrackSpeeds(BaseModel):
    """Normalised speed for each track, as sent to the motor driver."""

    model_config = _VALUE_OBJECT

    left: UnitSpeed
    right: UnitSpeed

    @classmethod
    def stopped(cls) -> TrackSpeeds:
        """Both tracks at rest."""
        return cls(left=0.0, right=0.0)

    @property
    def is_stopped(self) -> bool:
        return self.left == 0.0 and self.right == 0.0

    def scaled(self, factor: float) -> TrackSpeeds:
        """Return the speeds multiplied by ``factor`` (must be within [0, 1])."""
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"scale factor must be within [0, 1], got {factor}")
        return TrackSpeeds(left=self.left * factor, right=self.right * factor)


class DriveCommand(BaseModel):
    """Body-frame motion request for a differential (tracked) drive.

    Attributes:
        linear: Forward speed, normalised; positive drives forward.
        angular: Turn rate, normalised; positive turns counter-clockwise (left) seen from above.
    """

    model_config = _VALUE_OBJECT

    linear: UnitSpeed = 0.0
    angular: UnitSpeed = 0.0


class Frame(BaseModel):
    """A single BGR camera image with its capture time.

    Attributes:
        image: ``uint8`` array of shape ``(height, width, 3)`` in BGR channel order.
        timestamp_s: Monotonic capture time in seconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    image: NDArray[np.uint8]
    timestamp_s: float = Field(ge=0.0)

    @field_validator("image")
    @classmethod
    def _check_image(cls, value: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if not isinstance(value, np.ndarray):
            raise TypeError("image must be a numpy array")
        if value.dtype != np.uint8:
            raise ValueError(f"image dtype must be uint8, got {value.dtype}")
        if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"image must have shape (height, width, 3), got {value.shape}")
        return value

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])
