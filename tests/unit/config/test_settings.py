from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from jetank.config.settings import (
    CameraConfig,
    JetankSettings,
    MotorChannelConfig,
    MotorsConfig,
    load_settings,
)
from jetank.errors import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "jetank.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_safe() -> None:
    settings = load_settings()
    assert settings.motors.backend == "fake"
    assert settings.camera.backend == "fake"
    assert settings.drive.max_speed <= 0.5


def test_example_config_in_repo_is_valid() -> None:
    from pathlib import Path

    example = Path(__file__).parents[3] / "config" / "jetank.example.yaml"
    settings = load_settings(example)
    assert settings.motors.i2c_address == 0x60


def test_yaml_values_override_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, "motors:\n  i2c_bus: 1\ndrive:\n  max_speed: 0.3\n")
    settings = load_settings(path)
    assert settings.motors.i2c_bus == 1
    assert settings.drive.max_speed == 0.3
    assert settings.drive.watchdog_timeout_s == 0.5  # untouched default


def test_environment_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path, "motors:\n  i2c_bus: 1\n  i2c_address: 0x61\n")
    monkeypatch.setenv("JETANK_MOTORS__I2C_BUS", "7")
    settings = load_settings(path)
    assert settings.motors.i2c_bus == 7
    assert settings.motors.i2c_address == 0x61  # YAML value kept for keys not in env


def test_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert load_settings(write(tmp_path, "")) == JetankSettings()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("motors: [1, 2", "not valid YAML"),
        ("- just\n- a list\n", "mapping"),
        ("motors:\n  i2c_bus: -1\n", "invalid configuration"),
        ("unknown_section: {}\n", "invalid configuration"),
        ("motors:\n  backend: real\n", "invalid configuration"),
    ],
    ids=["syntax", "not-mapping", "range", "extra-key", "bad-backend"],
)
def test_bad_files_raise_configuration_error(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(write(tmp_path, text))


def test_missing_file_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_settings(tmp_path / "absent.yaml")


def test_motor_channel_needs_two_distinct_inputs() -> None:
    with pytest.raises(ValidationError, match="different channels"):
        MotorChannelConfig(pwm_in_a=3, pwm_in_b=3)


def test_motors_cannot_share_channels() -> None:
    with pytest.raises(ValidationError, match="share a PWM channel"):
        MotorsConfig(
            left=MotorChannelConfig(pwm_in_a=0, pwm_in_b=1),
            right=MotorChannelConfig(pwm_in_a=1, pwm_in_b=2),
        )


def test_camera_output_cannot_exceed_capture() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        CameraConfig(capture_width=640, capture_height=480, width=1280, height=720)
