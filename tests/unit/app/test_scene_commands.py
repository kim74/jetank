from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from jetank.app.cli import main
from jetank.domain.records import record_id_for
from jetank.domain.scenes import Lighting, SceneAction, SceneLabels, SceneMessage, SceneRecord
from jetank.drivers.scene_directory import SceneDirectory

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from jetank.control.teleop import TeleopSession

LABELS = SceneLabels(room="kitchen", lighting=Lighting.DAYLIGHT)


def test_scenes_without_records_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["scenes", "--directory", str(tmp_path / "none")]) == 0
    assert "no scenes" in capsys.readouterr().out


def test_scenes_counts_rooms_light_and_toys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SceneDirectory(tmp_path)
    labels = [
        SceneLabels(room="kitchen", lighting=Lighting.DAYLIGHT),
        SceneLabels(room="kitchen", lighting=Lighting.LAMP, has_animals=False),
        SceneLabels(room="living room", lighting=Lighting.LAMP),
    ]
    for second, label in enumerate(labels):
        moment = datetime(2026, 11, 22, 9, 0, second, tzinfo=timezone.utc)
        store.save(
            SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=label), b"x"
        )
    assert main(["scenes", "--directory", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "3 scenes" in out
    assert "  kitchen      2" in out
    assert "  living room  1" in out
    assert "  lamp      2" in out
    assert "  dim       0" in out
    assert "With animals: 2, without: 1" in out


def test_teleop_offers_scene_recording_and_the_configured_poses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jetank.app import teleop_command

    config = tmp_path / "jetank.yaml"
    config.write_text(
        "arm:\n  poses:\n    search: {positions: {shoulder: 724, camera_tilt: 572}}\n"
        f"scenes:\n  directory: {tmp_path / 'scenes'}\n",
        encoding="utf-8",
    )
    sessions: list[TeleopSession] = []

    async def interrupted_serve(self: teleop_command.TeleopServer, on_ready: object = None) -> None:
        sessions.append(self._session)  # type: ignore[attr-defined]
        raise KeyboardInterrupt

    monkeypatch.setattr(teleop_command.TeleopServer, "serve", interrupted_serve)
    assert main(["--config", str(config), "teleop"]) == 0
    status = sessions[0].connect("a")
    assert status.poses == ["search"]
    assert status.scenes is not None
    take = SceneMessage(type="scene", action=SceneAction.TAKE, labels=LABELS)
    error = sessions[0].handle("a", take).error
    assert error is not None
    assert "no camera image" in error  # nobody watches the video, so there is no frame
