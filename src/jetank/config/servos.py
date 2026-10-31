"""
Configuration of the serial servo bus and the gripper on it.

Layer: config
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from jetank.config.base import SECTION, Backend
from jetank.domain.servo import MAX_POSITION, GripperCalibration, JointCalibration, JointName


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
        max_torque: Torque cap as a fraction of the servo's maximum, applied before torque
            is switched on. Bounds the squeeze on a held object and the stall current.
    """

    model_config = SECTION

    servo_id: int = Field(default=4, ge=0, le=253)
    open_position: int = Field(default=745, ge=0, le=MAX_POSITION)
    closed_position: int = Field(default=965, ge=0, le=MAX_POSITION)
    move_time_s: float = Field(default=0.8, ge=0.2, le=5.0)
    max_torque: float = Field(default=0.5, ge=0.1, le=1.0)

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


def _default_joints() -> dict[JointName, JointCalibration]:
    # Measured 2026-10-31 by moving each limp joint by hand; limits are 20 counts inside the
    # comfortable ends and never beyond the servos' stored angle limits (20..1003).
    return {
        # horn re-fitted: left 90 deg 144, straight ahead ~575, right 90 deg 960 (monotonic)
        JointName.BASE: JointCalibration(
            servo_id=1, min_position=164, max_position=940, home_position=575
        ),
        # raised ~203 (lower number = up), lowered ~968, rest ~870
        JointName.SHOULDER: JointCalibration(
            servo_id=2, min_position=223, max_position=948, home_position=870
        ),
        # folded ~70, in line with the upper arm ~479, straight up ~862, extended ~1005
        # (higher number = up/back); the reading is blind just beyond both ends
        JointName.ELBOW: JointCalibration(
            servo_id=3, min_position=90, max_position=985, home_position=479
        ),
        # up ~224 (lower number = up), down ~803, level ~535
        JointName.CAMERA_TILT: JointCalibration(
            servo_id=5, min_position=244, max_position=783, home_position=535
        ),
    }


class ArmConfig(BaseModel):
    """Calibration of the arm joints. Joints not listed here can never be commanded.

    Attributes:
        joints: Calibration per joint.
        speed_cps: Joint speed in servo counts per second (~0.2 degrees per count).
    """

    model_config = SECTION

    joints: dict[JointName, JointCalibration] = Field(default_factory=_default_joints)
    speed_cps: float = Field(default=150.0, gt=0.0, le=1000.0)

    @model_validator(mode="after")
    def _one_servo_per_joint(self) -> ArmConfig:
        ids = [joint.servo_id for joint in self.joints.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("two joints use the same servo ID")
        return self
