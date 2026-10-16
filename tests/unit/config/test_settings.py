from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from jetank.config.settings import (
    CameraConfig,
    DriveConfig,
    JetankSettings,
    MotorChannelConfig,
    MotorsConfig,
    TeleopConfig,
    load_settings,
    unknown_environment_variables,
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


@pytest.mark.parametrize(
    ("pwm", "forward", "reverse"), [(3, 3, 4), (3, 4, 3), (4, 3, 3), (3, 3, 3)]
)
def test_motor_channel_needs_three_distinct_inputs(pwm: int, forward: int, reverse: int) -> None:
    with pytest.raises(ValidationError, match="three different channels"):
        MotorChannelConfig(pwm_channel=pwm, forward_channel=forward, reverse_channel=reverse)


def test_motors_cannot_share_channels() -> None:
    with pytest.raises(ValidationError, match=r"share PCA9685 channels: \[2\]"):
        MotorsConfig(
            left=MotorChannelConfig(pwm_channel=0, forward_channel=1, reverse_channel=2),
            right=MotorChannelConfig(pwm_channel=2, forward_channel=3, reverse_channel=4),
        )


def test_camera_output_cannot_exceed_capture() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        CameraConfig(capture_width=640, capture_height=480, width=1280, height=720)


def test_teleop_tick_must_be_well_below_watchdog_timeout() -> None:
    with pytest.raises(ValidationError, match="tick_period_s"):
        JetankSettings(
            drive=DriveConfig(watchdog_timeout_s=0.2), teleop=TeleopConfig(tick_period_s=0.1)
        )


def test_teleop_defaults_listen_on_the_network() -> None:
    teleop = JetankSettings().teleop
    assert (teleop.host, teleop.port) == ("0.0.0.0", 8080)


def test_misspelt_environment_variable_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Single underscore instead of the "__" nesting delimiter: silently ignored before.
    monkeypatch.setenv("JETANK_MOTORS_BACKEND", "hardware")
    with caplog.at_level("WARNING", logger="jetank.config.settings"):
        settings = load_settings()
    assert settings.motors.backend == "fake"
    assert "JETANK_MOTORS_BACKEND" in caplog.text
    assert "did you mean JETANK_MOTORS__BACKEND" in caplog.text


@pytest.mark.parametrize(
    "name",
    [
        "JETANK_MOTORS__BACKEND",
        "jetank_motors__backend",
        "JETANK_MOTORS__LEFT__TRIM",
        "JETANK_DRIVE",
        "JETANK_ALLOW_MOTION",
        "PATH",
    ],
)
def test_known_environment_variables_are_not_reported(name: str) -> None:
    assert unknown_environment_variables({name: "x"}) == []


def test_unknown_variable_without_close_match_is_reported_without_hint() -> None:
    assert unknown_environment_variables({"JETANK_ZZZ": "1"}) == ["JETANK_ZZZ"]
