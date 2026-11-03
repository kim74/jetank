from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from jetank.config.settings import (
    CameraConfig,
    DetectorConfig,
    DisplayConfig,
    DriveConfig,
    JetankSettings,
    MotorChannelConfig,
    MotorsConfig,
    TeleopConfig,
    load_settings,
    unknown_environment_variables,
)
from jetank.domain.detection import Response
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
    assert settings.display.backend == "fake"
    assert settings.power.backend == "fake"
    assert settings.drive.max_speed <= 0.5


def test_example_config_in_repo_is_valid() -> None:
    from pathlib import Path

    example = Path(__file__).parents[3] / "config" / "jetank.example.yaml"
    settings = load_settings(example)
    assert settings.motors.i2c_address == 0x60
    assert settings.display.i2c_address == 0x3C
    assert settings.power.i2c_address == 0x41


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


@pytest.mark.parametrize("height", [16, 48])
def test_display_height_must_be_a_supported_panel(height: int) -> None:
    with pytest.raises(ValidationError):
        DisplayConfig(height=height)  # type: ignore[arg-type]


def test_display_backend_can_be_set_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JETANK_DISPLAY__BACKEND", "hardware")
    assert load_settings().display.backend == "hardware"


def test_gripper_defaults_match_the_measured_jetank() -> None:
    settings = load_settings()
    assert settings.servos.backend == "fake"
    assert settings.servos.port == "/dev/ttyTHS1"
    assert settings.gripper.max_torque == 0.5
    assert settings.gripper.grip_overshoot == 130  # jaw play: they press only from ~655
    calibration = settings.gripper.calibration
    assert (calibration.servo_id, calibration.open_position, calibration.closed_position) == (
        4,
        955,
        740,
    )


def test_gripper_limits_too_close_are_rejected() -> None:
    from jetank.config.servos import GripperConfig

    with pytest.raises(ValidationError, match="20 counts apart"):
        GripperConfig(open_position=900, closed_position=905)


def test_grip_overshoot_reaches_the_calibration() -> None:
    from jetank.config.servos import GripperConfig

    config = GripperConfig(open_position=955, closed_position=740, grip_overshoot=40)
    assert config.calibration.grip_position == 700


def test_grip_overshoot_past_the_servo_range_is_rejected() -> None:
    from jetank.config.servos import GripperConfig

    with pytest.raises(ValidationError, match="grip position"):
        GripperConfig(open_position=500, closed_position=30, grip_overshoot=31)


def test_arm_defaults_cover_every_joint_and_stay_inside_servo_limits() -> None:
    from jetank.domain.servo import JointName

    joints = load_settings().arm.joints
    assert set(joints) == set(JointName)
    assert joints[JointName.ELBOW].servo_id == 3
    for joint in joints.values():
        assert 20 <= joint.min_position < joint.max_position <= 1003


def test_arm_joints_cannot_share_a_servo(tmp_path: Path) -> None:
    text = (
        "arm:\n  joints:\n"
        "    base: {servo_id: 1, min_position: 460, max_position: 983, home_position: 875}\n"
        "    shoulder: {servo_id: 1, min_position: 223, max_position: 948, home_position: 870}\n"
    )
    with pytest.raises(ConfigurationError, match="same servo ID"):
        load_settings(write(tmp_path, text))


def test_gripper_servo_cannot_be_an_arm_joint(tmp_path: Path) -> None:
    text = (
        "arm:\n  joints:\n"
        "    base: {servo_id: 4, min_position: 460, max_position: 983, home_position: 875}\n"
    )
    with pytest.raises(ConfigurationError, match="also configured as an arm joint"):
        load_settings(write(tmp_path, text))


def test_arm_jog_step_defaults_to_one_send_period_at_arm_speed() -> None:
    arm = load_settings().arm
    assert arm.jog_step == round(arm.speed_cps * 0.1)  # the page jogs every 100 ms


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_fps", 0.0),
        ("video_fps", 31.0),
        ("video_jpeg_quality", 5),
        ("video_jpeg_quality", 99),
    ],
)
def test_video_settings_out_of_range_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        TeleopConfig.model_validate({field: value})


def test_detector_responses_default_to_chasing_cats_and_reporting_people() -> None:
    detector = DetectorConfig()
    assert detector.response_for("cat") is Response.CHASE
    assert detector.response_for("person") is Response.ALERT
    assert detector.response_for("bicycle") is Response.IGNORE


def test_detector_responses_must_name_known_classes() -> None:
    with pytest.raises(ValidationError, match="does not know"):
        DetectorConfig.model_validate({"responses": {"hedgehog": "chase"}})


def test_detector_class_names_can_be_swapped_with_the_model() -> None:
    detector = DetectorConfig.model_validate(
        {"class_names": ["fox", "cat"], "responses": {"fox": "chase"}}
    )
    assert detector.response_for("fox") is Response.CHASE
    assert detector.response_for("cat") is Response.IGNORE
