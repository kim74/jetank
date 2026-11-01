from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from jetank.app.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_config_prints_resolved_settings(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["motors"]["backend"] == "fake"
    assert printed["motors"]["i2c_address"] == 0x60


def test_invalid_config_file_exits_with_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("drive:\n  max_speed: 5\n", encoding="utf-8")
    assert main(["--config", str(bad), "config"]) == 1
    assert "invalid configuration" in capsys.readouterr().err


def test_snapshot_with_fake_camera_writes_image(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    output = tmp_path / "frame.png"
    assert main(["snapshot", str(output), "--warmup", "2"]) == 0
    image = cv2.imread(str(output))
    assert image.shape == (480, 640, 3)


def test_snapshot_rejects_negative_warmup(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["snapshot", "x.jpg", "--warmup", "-1"]) == 1
    assert "warmup" in capsys.readouterr().err


def test_snapshot_unwritable_path_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    assert main(["snapshot", str(tmp_path / "frame.unknown"), "--warmup", "0"]) == 1
    assert "could not write image" in capsys.readouterr().err


def test_teleop_stops_and_closes_motors_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeMotorDriver

    motors = FakeMotorDriver()
    served: list[bool] = []

    async def interrupted_serve(self: object, on_ready: object = None) -> None:
        served.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_motor_driver", lambda _config: motors)
    monkeypatch.setattr(cli.TeleopServer, "serve", interrupted_serve)
    assert main(["teleop"]) == 0
    assert served == [True]
    assert motors.closed
    assert motors.current.is_stopped


def test_teleop_warns_when_motors_are_fake(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from jetank.app import cli

    async def interrupted_serve(self: object, on_ready: object = None) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.TeleopServer, "serve", interrupted_serve)
    with caplog.at_level("INFO"):
        assert main(["teleop"]) == 0
    assert "FAKE motors" in caplog.text
    assert "will not move" in caplog.text


def test_display_shows_status_until_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetank.app import cli, status_display
    from jetank.drivers.fake import FakeDisplay

    display = FakeDisplay()

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_display", lambda _config: display)
    monkeypatch.setattr(status_display.time, "sleep", interrupt)
    assert main(["display"]) == 0
    assert display.history[0].lines[-1] == "idle"
    assert display.closed
    assert display.current.lines == ()


def test_display_hardware_failure_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jetank.app import cli
    from jetank.errors import HardwareError

    def missing(_config: object) -> object:
        raise HardwareError("SSD1306 at 0x3C not responding")

    monkeypatch.setattr(cli, "build_display", missing)
    assert main(["display"]) == 1
    assert "0x3C not responding" in capsys.readouterr().err


def test_teleop_shows_mode_and_stops_motors_before_blanking_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from jetank.app import cli
    from jetank.drivers.fake import FakeDisplay, FakeMotorDriver

    events: list[str] = []
    motors = FakeMotorDriver()

    class RecordingDisplay(FakeDisplay):
        def close(self) -> None:
            events.append(f"display closed, motors closed={motors.closed}")
            super().close()

    display = RecordingDisplay()

    async def serve_until_status_shown(self: object, on_ready: object = None) -> None:
        while not display.history:
            await asyncio.sleep(0.001)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_motor_driver", lambda _config: motors)
    monkeypatch.setattr(cli, "build_display", lambda _config: display)
    monkeypatch.setattr(cli.TeleopServer, "serve", serve_until_status_shown)
    assert main(["teleop"]) == 0
    assert display.history[0].lines[-1] == "teleop: waiting"
    assert events == ["display closed, motors closed=True"]


def test_teleop_runs_without_a_working_display(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeMotorDriver
    from jetank.errors import HardwareError

    motors = FakeMotorDriver()
    served: list[bool] = []

    def missing(_config: object) -> object:
        raise HardwareError("SSD1306 at 0x3C not responding")

    async def interrupted_serve(self: object, on_ready: object = None) -> None:
        served.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_motor_driver", lambda _config: motors)
    monkeypatch.setattr(cli, "build_display", missing)
    monkeypatch.setattr(cli.TeleopServer, "serve", interrupted_serve)
    assert main(["teleop"]) == 0
    assert served == [True]
    assert motors.closed
    assert "continuing without it" in caplog.text


def test_gripper_status_is_read_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeServo, FakeServoBus

    bus = FakeServoBus(servos={4: FakeServo(position=848)})
    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: bus)
    assert main(["gripper", "status"]) == 0
    assert "opening 0.50, position 848" in capsys.readouterr().out
    assert bus.moves == []
    assert bus.torque_log == []
    assert bus.closed


@pytest.mark.parametrize(
    ("args", "target"), [(["open"], 955), (["close"], 610), (["set", "0.5"], 848)]
)
def test_gripper_moves_then_goes_limp(
    monkeypatch: pytest.MonkeyPatch, args: list[str], target: int
) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeServo, FakeServoBus

    bus = FakeServoBus(servos={4: FakeServo(position=900)})
    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: bus)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert main(["gripper", *args]) == 0
    assert bus.moves[-1].position == target
    assert bus.torque_log[-1] == (4, False)
    assert bus.closed


def test_gripper_interrupted_move_still_goes_limp(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeServo, FakeServoBus

    bus = FakeServoBus(servos={4: FakeServo(position=900)})

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: bus)
    monkeypatch.setattr(cli.time, "sleep", interrupt)
    assert main(["gripper", "open"]) == 0
    assert bus.torque_log[-1] == (4, False)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["set"], "needed with 'set'"),
        (["open", "0.5"], "only with 'set'"),
        (["set", "1.5"], "0.0-1.0"),
    ],
)
def test_gripper_rejects_bad_arguments(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["gripper", *args]) == 1
    assert message in capsys.readouterr().err


def test_gripper_missing_servo_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jetank.app import cli
    from jetank.drivers.fake import FakeServoBus

    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: FakeServoBus())
    assert main(["gripper", "status"]) == 1
    assert "servo 4 does not answer" in capsys.readouterr().err


def test_teleop_gripper_goes_limp_after_motors_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetank.app import cli
    from jetank.domain.teleop import GripperAction, GripperMessage
    from jetank.drivers.fake import FakeMotorDriver, FakeServo, FakeServoBus

    motors = FakeMotorDriver()
    events: list[str] = []

    class RecordingBus(FakeServoBus):
        def set_torque(self, servo_id: int, enabled: bool) -> None:
            if not enabled:
                events.append(f"gripper limp, motors closed={motors.closed}")
            super().set_torque(servo_id, enabled)

    bus = RecordingBus(servos={4: FakeServo(position=900)})

    async def serve_and_grip(self: cli.TeleopServer, on_ready: object = None) -> None:
        session = self._session  # type: ignore[attr-defined]
        session.connect("a")
        session.handle("a", GripperMessage(type="gripper", action=GripperAction.CLOSE))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_motor_driver", lambda _config: motors)
    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: bus)
    monkeypatch.setattr(cli.TeleopServer, "serve", serve_and_grip)
    assert main(["teleop"]) == 0
    assert bus.moves[-1].position == 610
    assert events == ["gripper limp, motors closed=True"]
    assert bus.closed


def test_teleop_runs_without_a_servo_bus(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from jetank.app import cli
    from jetank.errors import HardwareError

    def missing(_config: object) -> object:
        raise HardwareError("cannot open servo bus /dev/ttyTHS1")

    sessions: list[object] = []

    async def interrupted_serve(self: cli.TeleopServer, on_ready: object = None) -> None:
        sessions.append(self._session)  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_servo_bus", missing)
    monkeypatch.setattr(cli.TeleopServer, "serve", interrupted_serve)
    assert main(["teleop"]) == 0
    assert "without gripper and arm" in caplog.text
    status = sessions[0].status("x")  # type: ignore[attr-defined]
    assert status.gripper is None
    assert status.arm is None


def test_teleop_arm_jogs_and_goes_limp_after_motors_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    from jetank.app import cli
    from jetank.domain.servo import JointName
    from jetank.domain.teleop import ArmJogMessage
    from jetank.drivers.fake import FakeMotorDriver, FakeServo, FakeServoBus

    motors = FakeMotorDriver()
    events: list[str] = []

    class RecordingBus(FakeServoBus):
        def set_torque(self, servo_id: int, enabled: bool) -> None:
            if not enabled:
                events.append(f"servo {servo_id} limp, motors closed={motors.closed}")
            super().set_torque(servo_id, enabled)

    bus = RecordingBus(servos={3: FakeServo(position=479), 4: FakeServo(position=900)})

    async def serve_and_jog(self: cli.TeleopServer, on_ready: object = None) -> None:
        session = self._session  # type: ignore[attr-defined]
        session.connect("a")
        message = ArmJogMessage(type="arm", joint=JointName.ELBOW, direction=1)
        assert session.handle("a", message).error is None
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_motor_driver", lambda _config: motors)
    monkeypatch.setattr(cli, "build_servo_bus", lambda _config: bus)
    monkeypatch.setattr(cli.TeleopServer, "serve", serve_and_jog)
    assert main(["teleop"]) == 0
    assert bus.moves[-1].position == 479 + 15  # default jog step, elbow extends
    assert events == ["servo 3 limp, motors closed=True"]
    assert bus.closed
