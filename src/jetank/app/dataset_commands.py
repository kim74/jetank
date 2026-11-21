"""
``jetank scenes export`` and ``jetank dataset build``: from recorded scenes to training data.

Layer: app

1. ``jetank scenes export`` copies the scene images into the labelling folder, with a
   ``classes.txt`` for the labelling tool. Run it again after recording more scenes: only new
   images are added, nothing drawn so far is touched.
2. Outline toys and the floor in X-AnyLabeling (on a PC: copy the folder there and back).
3. ``jetank dataset build`` checks the drawings against the scenes and writes the YOLO
   segmentation dataset. With any error in the drawings nothing is written.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from jetank.domain.dataset import (
    DatasetClass,
    LabelIssue,
    Severity,
    Split,
    check_drawing,
    split_scenes,
    yolo_lines,
)
from jetank.drivers.labelling_directory import CLASSES_FILE, LabellingDirectory
from jetank.drivers.scene_directory import SceneDirectory
from jetank.drivers.yolo_dataset import DATA_FILE, YoloDatasetDirectory
from jetank.errors import JetankError

if TYPE_CHECKING:
    from pathlib import Path

    from jetank.config.dataset import DatasetConfig
    from jetank.config.settings import JetankSettings
    from jetank.interfaces.scene_store import SceneStore

logger = logging.getLogger(__name__)

#: Issues printed before the rest are only counted.
MAX_ISSUES_SHOWN: Final = 40
CLASS_NAMES: Final = tuple(c.value for c in DatasetClass)


class ExportReport(BaseModel):
    """What ``jetank scenes export`` did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: int
    present: int
    without_image: int


class DatasetReport(BaseModel):
    """What ``jetank dataset build`` found and wrote.

    Attributes:
        images: Images written per split (nothing is written when there are errors).
        outlines: Outlines written per class.
        unlabelled: Scenes without a drawing, left out.
        issues: Errors and warnings, by scene.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    images: dict[Split, int]
    outlines: dict[DatasetClass, int]
    unlabelled: int
    issues: list[LabelIssue]

    @property
    def errors(self) -> list[LabelIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]


def export_scenes(store: SceneStore, labelling: LabellingDirectory) -> ExportReport:
    """Copy every scene image that is not there yet into the labelling folder.

    Raises:
        StorageError: A scene or the folder cannot be read or written.
    """
    labelling.write_classes(CLASS_NAMES)
    added = present = without_image = 0
    for record in store.load_all():
        jpeg = store.load_image(record.record_id)
        if jpeg is None:
            without_image += 1
        elif labelling.add_image(record.record_id, jpeg):
            added += 1
        else:
            present += 1
    return ExportReport(added=added, present=present, without_image=without_image)


def build_dataset(
    store: SceneStore,
    labelling: LabellingDirectory,
    dataset: YoloDatasetDirectory,
    config: DatasetConfig,
) -> DatasetReport:
    """Check the drawings and, if there is no error, write the dataset.

    Raises:
        StorageError: Scenes, drawings or the dataset folder cannot be read or written.
    """
    scenes = {record.record_id: record for record in store.load_all()}
    drawings = labelling.load_drawings()
    issues = [
        issue for drawn in drawings for issue in check_drawing(drawn, scenes.get(drawn.record_id))
    ]
    labelled = {drawn.record_id for drawn in drawings}
    unlabelled = sum(record_id not in labelled for record_id in scenes)
    images = dict.fromkeys(Split, 0)
    outlines = dict.fromkeys(DatasetClass, 0)
    if not drawings or any(issue.severity is Severity.ERROR for issue in issues):
        return DatasetReport(images=images, outlines=outlines, unlabelled=unlabelled, issues=issues)

    # All scenes, not only the labelled ones: see split_scenes.
    splits = split_scenes(
        scenes.values(),
        val_fraction=config.val_fraction,
        block_s=config.block_s,
        run_gap_s=config.run_gap_s,
    )
    dataset.reset(CLASS_NAMES)
    for drawn in drawings:
        jpeg = store.load_image(drawn.record_id)
        if jpeg is None:
            issues.append(
                LabelIssue(
                    record_id=drawn.record_id,
                    severity=Severity.WARNING,
                    message="the scene has no image any more, left out",
                )
            )
            continue
        split = splits[drawn.record_id]
        dataset.add(drawn.record_id, split, jpeg, yolo_lines(drawn))
        images[split] += 1
        for shape in drawn.shapes:
            outlines[DatasetClass(shape.label)] += 1
    return DatasetReport(images=images, outlines=outlines, unlabelled=unlabelled, issues=issues)


def cmd_export_scenes(settings: JetankSettings, scenes: Path | None, to: Path | None) -> int:
    """``jetank scenes export``: put the scene images where the labelling tool can open them."""
    store = SceneDirectory(scenes if scenes is not None else settings.scenes.directory)
    if not store.directory.is_dir():
        raise JetankError(f"no scenes directory at {store.directory}")
    labelling = LabellingDirectory(to if to is not None else settings.dataset.labelling_directory)
    report = export_scenes(store, labelling)
    print(
        f"{report.added} new images in {labelling.directory} "
        f"({report.present} were there already, {report.without_image} scenes have no image)"
    )
    print(
        f"Open the folder in X-AnyLabeling, load the labels from {CLASSES_FILE} and outline "
        f"{' and '.join(CLASS_NAMES)} (polygons or rectangles), then run `jetank dataset build`."
    )
    return 0


def cmd_build_dataset(
    settings: JetankSettings, scenes: Path | None, labelling: Path | None, to: Path | None
) -> int:
    """``jetank dataset build``: check the drawings and write the YOLO segmentation dataset.

    Returns:
        0 when the dataset was written, 1 when the drawings have errors.
    """
    store = SceneDirectory(scenes if scenes is not None else settings.scenes.directory)
    drawings = LabellingDirectory(
        labelling if labelling is not None else settings.dataset.labelling_directory
    )
    if not drawings.directory.is_dir():
        raise JetankError(
            f"no labelling folder at {drawings.directory}; run `jetank scenes export` first"
        )
    dataset = YoloDatasetDirectory(to if to is not None else settings.dataset.directory)
    report = build_dataset(store, drawings, dataset, settings.dataset)
    for line in format_issues(report.issues):
        print(line, file=sys.stderr if report.errors else sys.stdout)
    if report.errors:
        print(f"{len(report.errors)} errors in the drawings: nothing written", file=sys.stderr)
        return 1
    if not sum(report.images.values()):
        raise JetankError(
            f"no labelled scenes in {drawings.directory} yet: nothing to train with "
            f"({report.unlabelled} scenes are waiting to be labelled)"
        )
    for line in format_report(report):
        print(line)
    print(f"Train with: yolo segment train data={dataset.directory / DATA_FILE}")
    return 0


def format_issues(issues: list[LabelIssue]) -> list[str]:
    """The issues as lines, errors first, at most ``MAX_ISSUES_SHOWN`` of them."""
    ordered = sorted(issues, key=lambda i: (i.severity is not Severity.ERROR, i.record_id))
    lines = [f"{i.record_id}  {i.severity.value}: {i.message}" for i in ordered]
    if len(lines) > MAX_ISSUES_SHOWN:
        lines = [*lines[:MAX_ISSUES_SHOWN], f"... and {len(lines) - MAX_ISSUES_SHOWN} more"]
    return lines


def format_report(report: DatasetReport) -> list[str]:
    """What was written, as lines."""
    total = sum(report.images.values())
    warnings = len(report.issues) - len(report.errors)
    return [
        f"{total} images: {report.images[Split.TRAIN]} train, {report.images[Split.VAL]} val",
        "Outlines: " + ", ".join(f"{cls.value} {count}" for cls, count in report.outlines.items()),
        f"{report.unlabelled} scenes not labelled yet (left out), {warnings} warnings",
    ]
