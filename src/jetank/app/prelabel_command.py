"""
``jetank scenes prelabel``: let the current model propose drawings for unlabelled scenes.

Layer: app

For every image in the labelling folder that has no drawing yet, the configured camera model
outlines the floor and boxes the toy animals, and the result is saved as a draft the labelling
tool opens like any drawing. Correcting a draft is quicker than drawing from scratch. Drafts
nobody has reviewed are left out of ``jetank dataset build`` (see
:mod:`jetank.drivers.labelling_directory`), so the model never trains on its own mistakes.

A low confidence threshold suits drafts: deleting a wrong box is quicker than drawing a
missed one.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from jetank.app.wiring import build_frame_analyzer
from jetank.domain.dataset import DatasetClass, draft_drawing
from jetank.domain.models import Frame
from jetank.drivers.labelling_directory import LabellingDirectory
from jetank.errors import JetankError
from jetank.perception.polygons import mask_polygons

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.interfaces.analyzer import FrameAnalyzer

logger = logging.getLogger(__name__)

#: Confidence threshold for drafts, lower than for driving.
DEFAULT_DRAFT_CONFIDENCE: Final = 0.25


class PrelabelReport(BaseModel):
    """What ``jetank scenes prelabel`` did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drafted: int
    already_drawn: int
    toys: int
    floor_outlines: int


def prelabel(analyzer: FrameAnalyzer, labelling: LabellingDirectory) -> PrelabelReport:
    """Draft a drawing for every image in ``labelling`` that has none.

    Raises:
        JetankError: An image cannot be read.
        StorageError: A draft cannot be written.
        InferenceError: The model fails.
    """
    import cv2

    drafted = already_drawn = toys = floors = 0
    for record_id in labelling.image_ids():
        if labelling.has_drawing(record_id):
            already_drawn += 1
            continue
        path = labelling.directory / f"{record_id}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise JetankError(f"could not read the image {path}")
        height, width = image.shape[:2]
        analysis = analyzer.analyze(Frame(image=image, timestamp_s=time.monotonic()))
        polygons = (
            mask_polygons(analysis.floor.cells, width, height) if analysis.floor is not None else []
        )
        draft = draft_drawing(record_id, analysis.detections, polygons, width, height)
        if labelling.save_draft(draft):
            drafted += 1
            toys += sum(s.label == DatasetClass.TOY_ANIMAL.value for s in draft.shapes)
            floors += sum(s.label == DatasetClass.FLOOR.value for s in draft.shapes)
        else:  # drawn in the tool meanwhile
            already_drawn += 1
    return PrelabelReport(
        drafted=drafted, already_drawn=already_drawn, toys=toys, floor_outlines=floors
    )


def cmd_prelabel(settings: JetankSettings, to: Path | None, confidence: float) -> int:
    """``jetank scenes prelabel``: draft drawings with the configured model.

    Raises:
        JetankError: No labelling folder, an unusable threshold, or the model cannot run.
    """
    if not 0.0 < confidence < 1.0:
        raise JetankError(f"--confidence must be between 0 and 1, not {confidence}")
    labelling = LabellingDirectory(to if to is not None else settings.dataset.labelling_directory)
    if not labelling.directory.is_dir():
        raise JetankError(
            f"no labelling folder at {labelling.directory}; run `jetank scenes export` first"
        )
    config = settings.detector.model_copy(update={"confidence_threshold": confidence})
    with contextlib.closing(build_frame_analyzer(config)) as analyzer:
        report = prelabel(analyzer, labelling)
    print(
        f"drafted {report.drafted} scenes ({report.toys} toy boxes, {report.floor_outlines} "
        f"floor outlines); {report.already_drawn} already had drawings and were left alone"
    )
    print(
        "Correct the drafts in X-AnyLabeling and save them (or tick a perfect one as checked): "
        "`jetank dataset build` leaves out drafts nobody has reviewed."
    )
    return 0
