"""
Training data for the toy and floor detector: drawn outlines, checks, train/val split.

Layer: domain

The operator draws outlines on the recorded scenes in a labelling tool (X-AnyLabeling): toy
animals and the drivable floor, as polygons or rectangles. This module turns what was drawn
into YOLO segmentation labels, checks it against the labels given while recording (a scene
"with toy animals" should have one outlined), and decides which scenes are used for
validation.

The split keeps neighbouring frames together. Automatic recording takes a frame every few
seconds, so two neighbours look almost the same; if one were in training and the other in
validation, the validation score would flatter the model. Each recording run is cut into
blocks of a few seconds and whole blocks go to validation, evenly spread over every run,
so each room and kind of scene is represented on both sides. Blocks are spans of recording
time, not counts of scenes, so the split does not depend on how densely a run was labelled.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from jetank.domain.records import RecordId

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from jetank.domain.scenes import SceneRecord

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")

#: Share of the scenes used for validation.
DEFAULT_VAL_FRACTION: Final = 0.2
#: Seconds of recording kept together on one side of the split.
DEFAULT_BLOCK_S: Final = 20.0
#: A pause longer than this between two scenes starts a new recording run.
DEFAULT_RUN_GAP_S: Final = 30.0
#: Decimal places of a YOLO coordinate (0.1 px on a 640 px image needs 5).
_DECIMALS: Final = 6


class DatasetClass(str, Enum):
    """What an outline marks. The order is the class index the model is trained with."""

    TOY_ANIMAL = "toy_animal"
    FLOOR = "floor"

    @property
    def class_index(self) -> int:
        return list(DatasetClass).index(self)


class Split(str, Enum):
    """Which part of the dataset a scene goes to."""

    TRAIN = "train"
    VAL = "val"


class Severity(str, Enum):
    """``error``: the dataset is not built. ``warning``: built, but worth a look."""

    ERROR = "error"
    WARNING = "warning"


class DrawnShape(BaseModel):
    """One shape as the labelling tool saved it.

    Attributes:
        label: The class name typed or picked in the tool.
        kind: The tool's shape type, e.g. "polygon" or "rectangle".
        points: Corners in pixels, ``(x, y)`` from the top left. A rectangle has two opposite
            corners (or all four).
    """

    model_config = _VALUE_OBJECT

    label: str
    kind: str
    points: tuple[tuple[float, float], ...]


class DrawnImage(BaseModel):
    """Everything drawn on one scene image.

    Attributes:
        record_id: The scene the image belongs to.
        width_px / height_px: Image size the points refer to.
        shapes: The outlines, possibly none (a saved image with nothing on it).
    """

    model_config = _VALUE_OBJECT

    record_id: RecordId
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    shapes: tuple[DrawnShape, ...] = ()


class LabelIssue(BaseModel):
    """Something wrong or suspicious in the drawn labels of one scene."""

    model_config = _VALUE_OBJECT

    record_id: str
    severity: Severity
    message: str


def check_drawing(drawn: DrawnImage, scene: SceneRecord | None) -> list[LabelIssue]:
    """Problems with what was drawn on one image.

    Errors: no recorded scene for the image, a class name that is not a :class:`DatasetClass`,
    a shape type other than polygon or rectangle, too few points. Warnings: the scene was
    recorded "with toy animals" but none is outlined (or the other way round), no floor
    outlined.

    Args:
        drawn: What was drawn.
        scene: The recorded scene with the same ID, or ``None`` if there is none.
    """
    issues: list[LabelIssue] = []

    def report(severity: Severity, message: str) -> None:
        issues.append(LabelIssue(record_id=drawn.record_id, severity=severity, message=message))

    if scene is None:
        report(Severity.ERROR, "no recorded scene with this ID (deleted in the scene review?)")
    names = {c.value for c in DatasetClass}
    for number, shape in enumerate(drawn.shapes, start=1):
        where = f"shape {number} ({shape.label!r})"
        if shape.label not in names:
            report(Severity.ERROR, f"{where}: unknown class, use one of {sorted(names)}")
        if shape.kind == "polygon" and len(shape.points) < 3:
            report(Severity.ERROR, f"{where}: a polygon needs at least 3 points")
        elif shape.kind == "rectangle" and len(shape.points) not in (2, 4):
            report(Severity.ERROR, f"{where}: a rectangle needs 2 or 4 points")
        elif shape.kind not in ("polygon", "rectangle"):
            report(Severity.ERROR, f"{where}: {shape.kind!r} shapes are not supported")
    labels = {shape.label for shape in drawn.shapes}
    if scene is not None:
        toys = DatasetClass.TOY_ANIMAL.value in labels
        if scene.labels.has_animals and not toys:
            report(Severity.WARNING, "recorded with toy animals, but none is outlined")
        if toys and not scene.labels.has_animals:
            report(Severity.WARNING, "toy animal outlined, but recorded without toy animals")
    if DatasetClass.FLOOR.value not in labels:
        report(Severity.WARNING, "no floor outlined")
    return issues


def yolo_lines(drawn: DrawnImage) -> list[str]:
    """The YOLO segmentation label file of a checked image, one line per shape.

    Each line is the class index followed by the polygon's corners as ``x y`` pairs, scaled
    to 0..1 by the image size. Points a little outside the image (drawn on its edge) are
    moved onto the edge; a rectangle becomes its four corners.

    Raises:
        ValueError: A shape fails :func:`check_drawing` (unknown class or unusable shape).
    """
    lines = []
    for shape in drawn.shapes:
        index = DatasetClass(shape.label).class_index
        corners = _corners(shape)
        coordinates = []
        for x, y in corners:
            coordinates.append(_unit(x / drawn.width_px))
            coordinates.append(_unit(y / drawn.height_px))
        lines.append(" ".join([str(index), *coordinates]))
    return lines


def _corners(shape: DrawnShape) -> Sequence[tuple[float, float]]:
    if shape.kind == "polygon" and len(shape.points) >= 3:
        return shape.points
    if shape.kind == "rectangle" and len(shape.points) in (2, 4):
        xs = [x for x, _ in shape.points]
        ys = [y for _, y in shape.points]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        return [(left, top), (right, top), (right, bottom), (left, bottom)]
    raise ValueError(f"unusable {shape.kind} with {len(shape.points)} points")


def _unit(value: float) -> str:
    return f"{min(max(value, 0.0), 1.0):.{_DECIMALS}f}"


def recording_runs(
    records: Iterable[SceneRecord], gap_s: float = DEFAULT_RUN_GAP_S
) -> list[list[SceneRecord]]:
    """The scenes grouped into recording runs, oldest first.

    A new run starts after a pause longer than ``gap_s`` or when the room changes.
    """
    runs: list[list[SceneRecord]] = []
    previous: SceneRecord | None = None
    for record in sorted(records, key=lambda r: r.recorded_at):
        pause_s = (
            math.inf
            if previous is None
            else (record.recorded_at - previous.recorded_at).total_seconds()
        )
        if previous is None or pause_s > gap_s or record.labels.room != previous.labels.room:
            runs.append([])
        runs[-1].append(record)
        previous = record
    return runs


def split_scenes(
    records: Iterable[SceneRecord],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    block_s: float = DEFAULT_BLOCK_S,
    run_gap_s: float = DEFAULT_RUN_GAP_S,
) -> dict[str, Split]:
    """Which scenes are for training and which for validation.

    Every run (see :func:`recording_runs`) is cut into blocks of ``block_s`` seconds from its
    first scene; a share ``val_fraction`` of each run's blocks, evenly spaced, goes to
    validation. Pass *all* recorded scenes, not only the labelled ones: then a scene's side
    depends only on when it was recorded, labelling more scenes never moves one across, and
    a run labelled sparsely still has scenes on both sides.

    Args:
        records: The recorded scenes.
        val_fraction: Share of blocks for validation, 0..1.
        block_s: Length of a block in seconds of recording, more than 0.
        run_gap_s: Pause that starts a new run.

    Returns:
        The split of every scene, by record ID.

    Raises:
        ValueError: ``val_fraction`` or ``block_s`` is out of range.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be within 0..1, not {val_fraction}")
    if not block_s > 0.0:
        raise ValueError(f"block_s must be more than 0, not {block_s}")
    splits: dict[str, Split] = {}
    for run in recording_runs(records, run_gap_s):
        for record in run:
            elapsed_s = (record.recorded_at - run[0].recorded_at).total_seconds()
            block = math.floor(elapsed_s / block_s)
            # Block i goes to validation where round((i + 1) * f) > round(i * f), rounding
            # half up: this spreads the validation blocks evenly over the run.
            to_val = math.floor((block + 1) * val_fraction + 0.5) > math.floor(
                block * val_fraction + 0.5
            )
            splits[record.record_id] = Split.VAL if to_val else Split.TRAIN
    return splits
