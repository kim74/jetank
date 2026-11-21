"""
Configuration of what the teleop page records: taught grasp poses and scenes for training.

Layer: config
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from jetank.config.base import SECTION


class TeachConfig(BaseModel):
    """Recording of taught grasp poses from the teleop page.

    Attributes:
        directory: Where grasp records (JSON) and their camera images (JPEG) are written.
            Relative paths are resolved from the working directory. The images show your
            home, so keep the directory out of git (``data/`` is ignored).
    """

    model_config = SECTION

    directory: Path = Path("data/grasps")


class ScenesConfig(BaseModel):
    """Recording of camera scenes from the teleop page, to train the toy and floor detector.

    Attributes:
        directory: Where scenes (JSON) and their images (JPEG) are written. The images show
            your home, so keep the directory out of git (``data/`` is ignored).
        jpeg_quality: JPEG quality of the saved images; higher than the live video, because
            compression artefacts hurt training.
        auto_period_s: Seconds between scenes while recording automatically.
        auto_limit_s: Automatic recording stops by itself after this many seconds.
    """

    model_config = SECTION

    directory: Path = Path("data/scenes")
    jpeg_quality: int = Field(default=95, ge=50, le=100)
    auto_period_s: float = Field(default=2.0, ge=0.5, le=60.0)
    auto_limit_s: float = Field(default=300.0, gt=0.0, le=3600.0)

    @model_validator(mode="after")
    def _limit_covers_a_period(self) -> ScenesConfig:
        if self.auto_limit_s < self.auto_period_s:
            raise ValueError("scenes.auto_limit_s must be at least scenes.auto_period_s")
        return self
