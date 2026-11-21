"""
Decoding the raw output of an Ultralytics YOLO model (exported without NMS) into a
:class:`~jetank.domain.perception.FrameAnalysis`.

Layer: perception

Output layout, for ``N`` candidate boxes (8400 at 640 x 640) and ``C`` classes:

- detection model: one tensor ``(1, 4 + C, N)``;
- segmentation model: ``(1, 4 + C + 32, N)`` plus mask prototypes ``(1, 32, H, W)``
  (160 x 160 at 640 x 640).

The first four rows are box centre x, y, width and height in input pixels, the next ``C``
the class scores (already 0..1). A segmentation model adds 32 mask coefficients per box: an
instance's mask is ``sigmoid(coefficients @ prototypes)``, cropped to its box.

The floor class (if configured) is not reported as detections: the masks of all floor
instances are merged into one :class:`~jetank.domain.perception.FloorMask`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jetank.domain.detection import BoundingBox, Detection
from jetank.domain.perception import FloorMask, FrameAnalysis, NetworkOutputs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from jetank.perception.letterbox import Letterbox

#: Mask coefficients per box in a YOLO segmentation model.
MASK_COEFFICIENTS: Final = 32


class YoloSettings(BaseModel):
    """How to read a YOLO model's output.

    Attributes:
        class_names: Label of each class index, in the model's order.
        confidence_threshold: Boxes scoring below this are dropped.
        nms_iou_threshold: Of two boxes of the same class overlapping more than this, the
            weaker is dropped.
        mask_threshold: A mask cell above this (0..1) counts as part of the instance.
        floor_label: The class that marks the floor, or ``None`` if the model has none.
        max_detections: At most this many boxes are kept, best first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    class_names: tuple[str, ...] = Field(min_length=1)
    confidence_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)
    nms_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    mask_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    floor_label: str | None = None
    max_detections: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def _floor_is_a_class(self) -> YoloSettings:
        if self.floor_label is not None and self.floor_label not in self.class_names:
            raise ValueError(f"floor_label {self.floor_label!r} is not one of the class names")
        return self


def decode(
    outputs: NetworkOutputs, box: Letterbox, settings: YoloSettings, timestamp_s: float
) -> FrameAnalysis:
    """What the model found, in normalised frame coordinates.

    Args:
        outputs: The network's output tensors.
        box: How the frame was placed in the network input.
        settings: Class names and thresholds.
        timestamp_s: Capture time of the frame.

    Returns:
        The detections (best first, floor excluded) and, when the model segments and a
        floor class is set, the floor mask: all ``False`` if no floor was found.

    Raises:
        ValueError: The outputs do not fit the class list (wrong model for the config).
    """
    prediction = outputs.with_rank(3)
    protos = outputs.with_rank(4)[0] if outputs.has_rank(4) else None
    classes = len(settings.class_names)
    coefficients = MASK_COEFFICIENTS if protos is not None else 0
    if prediction.shape[0] != 1 or prediction.shape[1] != 4 + classes + coefficients:
        raise ValueError(
            f"output {prediction.shape} does not fit {classes} classes"
            + (" with masks" if protos is not None else "")
        )
    rows = prediction[0].T  # (N, 4 + C [+ 32])
    scores = rows[:, 4 : 4 + classes]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(rows)), class_ids]
    candidates = confidences >= settings.confidence_threshold
    rows, class_ids, confidences = rows[candidates], class_ids[candidates], confidences[candidates]
    corners = _corners(rows[:, :4])
    kept = _nms(corners, confidences, class_ids, settings.nms_iou_threshold)
    kept = kept[: settings.max_detections]

    floor_id = (
        settings.class_names.index(settings.floor_label)
        if settings.floor_label is not None
        else None
    )
    detections = []
    for i in kept:
        if class_ids[i] == floor_id:
            continue
        x, y = box.to_frame(corners[i, [0, 2]], corners[i, [1, 3]])
        detections.append(
            Detection(
                label=settings.class_names[class_ids[i]],
                # FP16 engines can round a score a hair above 1.
                confidence=min(float(confidences[i]), 1.0),
                box=BoundingBox(x1=float(x[0]), y1=float(y[0]), x2=float(x[1]), y2=float(y[1])),
            )
        )
    floor = None
    if protos is not None and floor_id is not None:
        floor_rows = [i for i in kept if class_ids[i] == floor_id]
        floor = _floor_mask(
            rows[floor_rows, 4 + classes :], corners[floor_rows], protos, box, settings
        )
    return FrameAnalysis(timestamp_s=timestamp_s, detections=tuple(detections), floor=floor)


def _corners(xywh: NDArray[np.float32]) -> NDArray[np.float32]:
    # A negative size would turn the corners round; treat it as an empty box.
    half_w, half_h = np.clip(xywh[:, 2], 0, None) / 2, np.clip(xywh[:, 3], 0, None) / 2
    return np.stack(
        [xywh[:, 0] - half_w, xywh[:, 1] - half_h, xywh[:, 0] + half_w, xywh[:, 1] + half_h],
        axis=1,
    )


def _nms(
    corners: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.int64],
    iou_threshold: float,
) -> list[int]:
    """Indices of the boxes kept, best first; boxes only suppress boxes of their own class."""
    # Shifting each class far away keeps classes from overlapping each other.
    shifted = corners + (class_ids.astype(np.float32) * 1e5)[:, np.newaxis]
    order = list(np.argsort(-scores, kind="stable"))
    areas = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    kept: list[int] = []
    while order:
        best = order.pop(0)
        kept.append(int(best))
        if not order:
            break
        rest = np.array(order)
        width = np.minimum(shifted[best, 2], shifted[rest, 2]) - np.maximum(
            shifted[best, 0], shifted[rest, 0]
        )
        height = np.minimum(shifted[best, 3], shifted[rest, 3]) - np.maximum(
            shifted[best, 1], shifted[rest, 1]
        )
        overlap = np.clip(width, 0, None) * np.clip(height, 0, None)
        union = areas[best] + areas[rest] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = [int(i) for i, value in zip(rest, iou, strict=True) if value <= iou_threshold]
    return kept


def _floor_mask(
    coefficients: NDArray[np.float32],
    corners: NDArray[np.float32],
    protos: NDArray[np.float32],
    box: Letterbox,
    settings: YoloSettings,
) -> FloorMask:
    channels, height, width = protos.shape
    union = np.zeros((height, width), dtype=np.bool_)
    if len(coefficients):
        logits = (coefficients @ protos.reshape(channels, -1)).reshape(-1, height, width)
        masks = 1.0 / (1.0 + np.exp(-logits)) > settings.mask_threshold
        # Crop each mask to its box, in prototype cells.
        sx, sy = width / box.input_width, height / box.input_height
        cols = np.arange(width)[np.newaxis, :]
        rows = np.arange(height)[:, np.newaxis]
        for mask, (x1, y1, x2, y2) in zip(masks, corners, strict=True):
            inside = (cols >= x1 * sx) & (cols < x2 * sx) & (rows >= y1 * sy) & (rows < y2 * sy)
            union |= mask & inside
    row_range, col_range = box.frame_region(width, height)
    return FloorMask(cells=np.ascontiguousarray(union[row_range, col_range]))
