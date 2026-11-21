"""
``jetank scenes``: how many scenes have been recorded, and how varied they are.

Layer: app
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jetank.domain.scenes import Lighting, summarize_scenes
from jetank.drivers.scene_directory import SceneDirectory

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.domain.scenes import SceneSummary


def cmd_scenes(settings: JetankSettings, directory: Path | None) -> int:
    """Print the number of scenes per room, per kind of light, and with or without toy animals."""
    source = directory if directory is not None else settings.scenes.directory
    records = SceneDirectory(source).load_all()
    if not records:
        print(f"no scenes in {source}")
        return 0
    print(f"{len(records)} scenes in {source}\n")
    for line in format_summary(summarize_scenes(records)):
        print(line)
    return 0


def format_summary(summary: SceneSummary) -> list[str]:
    """The summary as lines of text."""
    width = max(len(room) for room in summary.rooms)
    lines = ["Rooms:"]
    lines += [f"  {room:<{width}}  {count}" for room, count in summary.rooms.items()]
    lines.append("Light:")
    lines += [f"  {light.value:<8}  {summary.lighting.get(light, 0)}" for light in Lighting]
    lines.append(
        f"With animals: {summary.with_animals}, without: {summary.total - summary.with_animals}"
    )
    return lines
