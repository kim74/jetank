"""
Configuration of the serial servo bus and the gripper on it.

Layer: config
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from jetank.config.base import SECTION, Backend
from jetank.domain.servo import MAX_POSITION, GripperCalibration


class ServosConfig(BaseModel):
    """Feetech SCS serial bus (UART on header pins 8/10).

    Attributes:
        port: Serial device; the user must be in the ``dialout`` group.
        baud_rate: Bus speed; the JETANK servos ship at 1 Mbaud.
        response_timeout_s: How long to wait for one status reply.
        retries: Extra attempts after a missing or corrupt reply.
    """

    model_config = SECTION

    backend: Backend = "fake"
    port: str = "/dev/ttyTHS1"
    baud_rate: int = Field(default=1_000_000, gt=0)
    response_timeout_s: float = Field(default=0.02, gt=0.0, le=1.0)
    retries: int = Field(default=2, ge=0, le=5)


class GripperConfig(BaseModel):
    """Gripper servo and its software limits.

    Defaults were measured on 2026-10-30 by moving the unpowered gripper by hand
    (open ~734, closed ~975) and keeping ~10 counts away from both mechanical ends.

    Attributes:
        servo_id: Bus ID of the gripper servo.
        open_position / closed_position: Raw positions of the open and closed limits.
        move_time_s: Time for a full open-to-closed stroke; shorter moves are scaled down.
    """

    model_config = SECTION

    servo_id: int = Field(default=4, ge=0, le=253)
    open_position: int = Field(default=745, ge=0, le=MAX_POSITION)
    closed_position: int = Field(default=965, ge=0, le=MAX_POSITION)
    move_time_s: float = Field(default=0.8, ge=0.2, le=5.0)

    @model_validator(mode="after")
    def _valid_calibration(self) -> GripperConfig:
        self.calibration  # noqa: B018 - raises if the limits are unusable
        return self

    @property
    def calibration(self) -> GripperCalibration:
        return GripperCalibration(
            servo_id=self.servo_id,
            open_position=self.open_position,
            closed_position=self.closed_position,
        )
