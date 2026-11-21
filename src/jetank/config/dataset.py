"""
Configuration of the detector's training data: where it is labelled and built, and the split.

Layer: config
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jetank.config.base import SECTION
from jetank.domain.dataset import DEFAULT_BLOCK_S, DEFAULT_RUN_GAP_S, DEFAULT_VAL_FRACTION


class DatasetConfig(BaseModel):
    """Training data made from the recorded scenes.

    Attributes:
        labelling_directory: Where ``jetank scenes export`` puts the scene images for the
            labelling tool, and where the tool saves what is drawn (one JSON per image).
        directory: Where ``jetank dataset build`` writes the YOLO segmentation dataset.
        val_fraction: Share of the scenes used for validation.
        block_s: Seconds of recording kept together on one side of the split.
        run_gap_s: A pause longer than this between two scenes starts a new recording run.

    Both directories hold photos of your home: keep them out of git (``data/`` is ignored).
    """

    model_config = SECTION

    labelling_directory: Path = Path("data/labelling")
    directory: Path = Path("data/dataset")
    val_fraction: float = Field(default=DEFAULT_VAL_FRACTION, ge=0.0, le=0.5)
    block_s: float = Field(default=DEFAULT_BLOCK_S, gt=0.0, le=600.0)
    run_gap_s: float = Field(default=DEFAULT_RUN_GAP_S, gt=0.0)
