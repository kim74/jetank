from __future__ import annotations

import pytest

from jetank.app import wiring
from jetank.app.wiring import (
    FAKE_SERVO_IDS,
    build_display,
    build_power_monitor,
    build_robot,
    build_servo_bus,
)
from jetank.config.servos import ServosConfig
from jetank.config.settings import DisplayConfig, JetankSettings, PowerConfig
from jetank.domain.models import DriveCommand
from jetank.drivers.fake import (
    FakeCamera,
    FakeClock,
    FakeDisplay,
    FakeMotorDriver,
    FakePowerMonitor,
)
from jetank.errors import CameraError


def test_fake_backends_build_a_working_robot(clock: FakeClock) -> None:
    with build_robot(JetankSettings(), clock=clock) as robot:
        assert isinstance(robot.motors, FakeMotorDriver)
        assert isinstance(robot.camera, FakeCamera)
        robot.drive.command(DriveCommand(linear=1.0))
        frame = robot.camera.read()
        assert frame.width == JetankSettings().camera.width
    assert robot.motors.closed
    assert robot.motors.history[-1].is_stopped
    assert robot.camera.closed


def test_failure_while_wiring_closes_what_was_built(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    built: list[FakeMotorDriver] = []

    def motors_factory(_config: object) -> FakeMotorDriver:
        built.append(FakeMotorDriver())
        return built[-1]

    def broken_camera(_config: object, _clock: object) -> FakeCamera:
        raise CameraError("no camera")

    monkeypatch.setattr(wiring, "build_motor_driver", motors_factory)
    monkeypatch.setattr(wiring, "build_camera", broken_camera)
    with pytest.raises(CameraError):
        build_robot(JetankSettings(), clock=clock)
    assert built[0].closed


def test_robot_uses_system_clock_by_default() -> None:
    robot = build_robot(JetankSettings())
    try:
        assert robot.camera.read().timestamp_s > 0
    finally:
        robot.close()


def test_fake_display_matches_panel_height() -> None:
    display = build_display(DisplayConfig(height=64))
    assert isinstance(display, FakeDisplay)
    assert (display.columns, display.rows) == (21, 8)


def test_fake_power_monitor_is_built_by_default() -> None:
    assert isinstance(build_power_monitor(PowerConfig()), FakePowerMonitor)


def test_fake_servo_bus_simulates_the_jetank_servos() -> None:
    bus = build_servo_bus(ServosConfig())
    assert all(bus.ping(servo_id) for servo_id in FAKE_SERVO_IDS)
    assert not bus.ping(6)
