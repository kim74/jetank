from __future__ import annotations

import pytest

from jetank.app import wiring
from jetank.app.wiring import build_robot
from jetank.config.settings import JetankSettings
from jetank.domain.models import DriveCommand
from jetank.drivers.fake import FakeCamera, FakeClock, FakeMotorDriver
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
