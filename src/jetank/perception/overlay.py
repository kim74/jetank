"""
Drawing what the camera model found over the camera image: floor tinted, boxes outlined.

Layer: perception
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from jetank.domain.perception import FrameAnalysis

#: BGR colours.
FLOOR_TINT: Final = (60, 200, 60)
BOX_COLOUR: Final = (40, 40, 240)
_FLOOR_ALPHA: Final = 0.35


def draw_analysis(image_bgr: NDArray[np.uint8], analysis: FrameAnalysis) -> NDArray[np.uint8]:
    """A copy of ``image_bgr`` with the floor tinted green and each detection boxed in red,
    labelled with its class and confidence."""
    import cv2  # imported here like elsewhere in the package

    out = image_bgr.copy()
    height, width = out.shape[:2]
    if analysis.floor is not None:
        cells = analysis.floor.cells.astype(np.uint8)
        floor = cv2.resize(cells, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = np.array(FLOOR_TINT, dtype=np.float32)
        out[floor] = (out[floor] * (1 - _FLOOR_ALPHA) + tint * _FLOOR_ALPHA).astype(np.uint8)
    for detection in analysis.detections:
        b = detection.box
        top_left = (round(b.x1 * (width - 1)), round(b.y1 * (height - 1)))
        bottom_right = (round(b.x2 * (width - 1)), round(b.y2 * (height - 1)))
        cv2.rectangle(out, top_left, bottom_right, BOX_COLOUR, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            out,
            text,
            (top_left[0], max(top_left[1] - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOX_COLOUR,
            1,
            cv2.LINE_AA,
        )
    return out
