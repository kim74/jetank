"""
Runtime configuration: every hardware address, bus, limit and calibration value lives here.

Layer: config

Values are resolved in this order (highest priority first):
1. Environment variables, prefixed ``JETANK_`` with ``__`` between nested keys,
   e.g. ``JETANK_MOTORS__I2C_BUS=1``.
2. The YAML file passed to :func:`load_settings`.
3. The defaults below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from jetank.domain.coco import COCO_CLASSES
from jetank.domain.detection import Response
from jetank.errors import ConfigurationError

Backend = Literal["hardware", "fake"]

_SECTION = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class MotorChannelConfig(BaseModel):
    """Wiring and calibration of one track motor on the PCA9685 + TB6612 board.

    The TB6612 takes a PWM input for speed and two digital inputs for direction; each is fed
    by its own PCA9685 channel.

    Attributes:
        pwm_channel: PCA9685 channel driving the H-bridge PWM (speed) input.
        forward_channel: PCA9685 channel driven high to run the track forward.
        reverse_channel: PCA9685 channel driven high to run the track in reverse.
        trim: Gain applied to this motor to compensate for mismatched motors.
        inverted: Flip the direction, for a motor wired the other way round.
    """

    model_config = _SECTION

    pwm_channel: int = Field(ge=0, le=15)
    forward_channel: int = Field(ge=0, le=15)
    reverse_channel: int = Field(ge=0, le=15)
    trim: float = Field(default=1.0, gt=0.0, le=1.5)
    inverted: bool = False

    @property
    def channels(self) -> frozenset[int]:
        """All PCA9685 channels this motor uses."""
        return frozenset((self.pwm_channel, self.forward_channel, self.reverse_channel))

    @model_validator(mode="after")
    def _distinct_channels(self) -> MotorChannelConfig:
        if len(self.channels) != 3:
            raise ValueError("pwm, forward and reverse must be three different channels")
        return self


class MotorsConfig(BaseModel):
    """Track motor driver (PCA9685 PWM expander feeding a TB6612 H-bridge).

    Channel defaults are the Adafruit MotorHAT layout, verified on the JETANK on 2026-10-16
    by pulsing each track with the tracks lifted.
    """

    model_config = _SECTION

    backend: Backend = "fake"
    # Orin Nano 40-pin header pins 3/5, verified with `i2cdetect -y -r 7`.
    i2c_bus: int = Field(default=7, ge=0)
    i2c_address: int = Field(default=0x60, ge=0x03, le=0x77)
    # 1526 Hz is the chip maximum (prescale 3); the legacy code asked for 1600 and got this.
    pwm_frequency_hz: int = Field(default=1526, ge=24, le=1526)
    left: MotorChannelConfig = MotorChannelConfig(
        pwm_channel=8, forward_channel=9, reverse_channel=10
    )
    right: MotorChannelConfig = MotorChannelConfig(
        pwm_channel=13, forward_channel=12, reverse_channel=11
    )

    @model_validator(mode="after")
    def _no_shared_channels(self) -> MotorsConfig:
        shared = self.left.channels & self.right.channels
        if shared:
            raise ValueError(f"left and right motors share PCA9685 channels: {sorted(shared)}")
        return self


class DriveConfig(BaseModel):
    """Safety limits for the drive controller.

    Attributes:
        max_speed: Fraction of full motor power that a full-scale command maps to.
        watchdog_timeout_s: Motors stop if no command arrives within this time.
    """

    model_config = _SECTION

    max_speed: float = Field(default=0.5, gt=0.0, le=1.0)
    watchdog_timeout_s: float = Field(default=0.5, gt=0.0, le=10.0)


class TeleopConfig(BaseModel):
    """Remote-control web server (``jetank teleop``).

    There is no authentication: anyone who can reach the port can drive the robot, so only
    run it on a trusted network.

    Attributes:
        host: Interface to listen on; ``0.0.0.0`` accepts connections from the network.
        port: TCP port for the control page and its WebSocket; 0 picks a free port.
        tick_period_s: How often the drive watchdog is checked. Must be well below
            ``drive.watchdog_timeout_s``.
    """

    model_config = _SECTION

    host: str = "0.0.0.0"  # reachable from the LAN on purpose
    port: int = Field(default=8080, ge=0, le=65535)
    tick_period_s: float = Field(default=0.05, gt=0.0, le=1.0)


class CameraConfig(BaseModel):
    """CSI camera (IMX219) captured through GStreamer ``nvarguscamerasrc``.

    Attributes:
        sensor_id: CSI port index (0 = CAM0).
        capture_width / capture_height / fps: Sensor mode requested from Argus.
        width / height: Size of the frames delivered to the application.
        flip_method: ``nvvidconv`` flip method (0 = none, 2 = rotate 180°).
    """

    model_config = _SECTION

    backend: Backend = "fake"
    sensor_id: int = Field(default=0, ge=0, le=7)
    capture_width: int = Field(default=1640, gt=0)
    capture_height: int = Field(default=1232, gt=0)
    fps: int = Field(default=30, gt=0, le=120)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    flip_method: int = Field(default=0, ge=0, le=7)

    @model_validator(mode="after")
    def _output_not_larger_than_capture(self) -> CameraConfig:
        if self.width > self.capture_width or self.height > self.capture_height:
            raise ValueError("output size must not exceed capture size")
        return self


def _default_responses() -> dict[str, Response]:
    return {"cat": Response.CHASE, "dog": Response.CHASE, "person": Response.ALERT}


class DetectorConfig(BaseModel):
    """Object detector (YOLO exported to a TensorRT engine by ``scripts/export_detector.sh``).

    Attributes:
        backend: ``tensorrt`` runs the engine on the GPU; ``fake`` detects nothing.
        engine_path: Serialized TensorRT engine; relative paths are resolved from the
            working directory.
        class_names: Label for each output class index of the model, in order.
        confidence_threshold: Detections below this score are discarded.
        nms_iou_threshold: Overlapping boxes of the same class above this IoU are merged.
        responses: What to do about each label; labels not listed are ignored.
    """

    model_config = _SECTION

    backend: Literal["tensorrt", "fake"] = "fake"
    engine_path: Path = Path("models/yolo11n_640_fp16.engine")
    class_names: tuple[str, ...] = COCO_CLASSES
    confidence_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)
    nms_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    responses: dict[str, Response] = Field(default_factory=_default_responses)

    @model_validator(mode="after")
    def _responses_refer_to_known_classes(self) -> DetectorConfig:
        unknown = set(self.responses) - set(self.class_names)
        if unknown:
            raise ValueError(f"responses name labels the model does not know: {sorted(unknown)}")
        return self

    def response_for(self, label: str) -> Response:
        return self.responses.get(label, Response.IGNORE)


class TrackerConfig(BaseModel):
    """Frame-to-frame object tracker.

    Attributes:
        match_iou_threshold: Minimum overlap for a detection to continue an existing track.
        confirm_hits: Matches needed before a track counts as a real intruder (filters
            one-frame false positives).
        max_age_s: A track not matched for this long is dropped.
    """

    model_config = _SECTION

    match_iou_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    confirm_hits: int = Field(default=3, ge=1, le=100)
    max_age_s: float = Field(default=1.0, gt=0.0, le=60.0)


class EventsConfig(BaseModel):
    """Recording of confirmed intruders.

    Attributes:
        directory: Where annotated snapshots and JSON records are written.
        label_cooldown_s: After recording a label, further events of the same label are
            not recorded for this long (avoids floods when a track is lost and re-found).
    """

    model_config = _SECTION

    directory: Path = Path("events")
    label_cooldown_s: float = Field(default=30.0, ge=0.0)


class JetankSettings(BaseSettings):
    """Top-level configuration for the robot."""

    model_config = SettingsConfigDict(
        env_prefix="JETANK_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    motors: MotorsConfig = MotorsConfig()
    drive: DriveConfig = DriveConfig()
    teleop: TeleopConfig = TeleopConfig()
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    events: EventsConfig = EventsConfig()

    @model_validator(mode="after")
    def _watchdog_checked_often_enough(self) -> JetankSettings:
        if self.teleop.tick_period_s >= self.drive.watchdog_timeout_s / 2:
            raise ValueError(
                "teleop.tick_period_s must be less than half of drive.watchdog_timeout_s, "
                "or the watchdog stops the tracks late"
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment wins over init kwargs, which is where the YAML file contents arrive.
        return env_settings, init_settings


def load_settings(path: Path | None = None) -> JetankSettings:
    """Build settings from an optional YAML file plus ``JETANK_*`` environment variables.

    Args:
        path: YAML file to read; ``None`` uses defaults and environment only.

    Raises:
        ConfigurationError: The file is missing, is not a YAML mapping, or fails validation.
    """
    data = _read_yaml(path) if path is not None else {}
    try:
        return JetankSettings(**data)
    except ValidationError as exc:
        source = str(path) if path is not None else "defaults/environment"
        raise ConfigurationError(f"invalid configuration ({source}):\n{exc}") from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"config file {path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"config file {path} must contain a mapping at the top level")
    return data
