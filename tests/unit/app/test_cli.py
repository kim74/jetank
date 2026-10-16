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
