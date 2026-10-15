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
