"""
Serial bus servo readings and the gripper's calibration.

Layer: domain

Positions are raw servo counts (0-1023 on the Feetech SCS series, about 0.195 degrees each).
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")

#: Highest raw position an SCS servo accepts.
MAX_POSITION: Final = 1023
#: Smallest open-to-closed travel accepted as a calibration (about 4 degrees).
MIN_GRIPPER_SPAN: Final = 20

ServoId = Annotated[int, Field(ge=0, le=253, description="bus ID; 254 is broadcast")]
ServoPosition = Annotated[int, Field(ge=0, le=MAX_POSITION, description="raw counts")]
#: How far the gripper is open: 0.0 = closed limit, 1.0 = open limit.
Opening = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ServoState(BaseModel):
    """One reading of a servo's live registers.

    Attributes:
        servo_id: Bus ID of the servo.
        position: Present position in raw counts.
        voltage_v: Supply voltage at the servo, in volts.
        temperature_c: Internal temperature in degrees Celsius.
        moving: Whether the servo reports it is still moving.
        torque_enabled: Whether the servo is actively holding or driving its position.
        error_flags: Fault bits from the servo's status reply (0 = no fault).
    """

    model_config = _VALUE_OBJECT

    servo_id: ServoId
    position: ServoPosition
    voltage_v: float = Field(ge=0.0, le=25.5)
    temperature_c: int = Field(ge=0, le=255)
    moving: bool
    torque_enabled: bool
    error_flags: int = Field(default=0, ge=0, le=0xFF)


class GripperCalibration(BaseModel):
    """Raw servo positions at the gripper's software limits.

    Either direction is allowed: on the JETANK, opening lowers the position number.

    Attributes:
        servo_id: Bus ID of the gripper servo.
        open_position: Position at the open limit.
        closed_position: Position at the closed limit.
    """

    model_config = _VALUE_OBJECT

    servo_id: ServoId
    open_position: ServoPosition
    closed_position: ServoPosition

    @model_validator(mode="after")
    def _enough_travel(self) -> GripperCalibration:
        if abs(self.open_position - self.closed_position) < MIN_GRIPPER_SPAN:
            raise ValueError(
                f"open and closed positions must be at least {MIN_GRIPPER_SPAN} counts apart"
            )
        return self

    def target(self, opening: float) -> int:
        """Raw position for ``opening`` (0.0 = closed, 1.0 = open); always within the limits.

        Raises:
            ValueError: ``opening`` is outside [0, 1] or not a number.
        """
        if not 0.0 <= opening <= 1.0:  # also rejects NaN
            raise ValueError(f"opening must be within [0, 1], got {opening}")
        span = self.open_position - self.closed_position
        return round(self.closed_position + opening * span)

    def opening(self, position: int) -> float:
        """Opening corresponding to ``position``, clamped to [0, 1] when beyond the limits."""
        span = self.open_position - self.closed_position
        return max(0.0, min(1.0, (position - self.closed_position) / span))

    def clamp(self, position: int) -> int:
        """``position`` moved to the nearest limit if it lies outside them."""
        low, high = sorted((self.open_position, self.closed_position))
        return max(low, min(high, position))


class GripperStatus(BaseModel):
    """What the gripper is doing.

    Attributes:
        opening: Current opening (0.0 = closed, 1.0 = open), clamped to the limits.
        servo: The underlying servo reading.
    """

    model_config = _VALUE_OBJECT

    opening: Opening
    servo: ServoState
