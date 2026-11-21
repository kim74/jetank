from __future__ import annotations

import numpy as np
import pytest

from jetank.domain.detection import BoundingBox, Detection
from jetank.domain.perception import FloorMask, FrameAnalysis
from jetank.perception.overlay import BOX_COLOUR, draw_analysis

pytest.importorskip("cv2")


def test_floor_is_tinted_and_boxes_are_drawn_on_a_copy() -> None:
    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    cells = np.zeros((120, 160), dtype=bool)
    cells[60:] = True
    analysis = FrameAnalysis(
        timestamp_s=0.0,
        detections=(
            Detection(
                label="toy_animal", confidence=0.9, box=BoundingBox(x1=0.1, y1=0.1, x2=0.3, y2=0.3)
            ),
        ),
        floor=FloorMask(cells=cells),
    )
    out = draw_analysis(image, analysis)
    assert (image == 100).all()  # the original is untouched
    assert tuple(out[10, 10]) == (100, 100, 100)  # neither floor nor box
    assert out[400, 320, 1] > 100  # floor: greener
    assert tuple(out[round(0.1 * 479), round(0.2 * 639)]) == BOX_COLOUR  # top edge of the box


def test_nothing_found_leaves_the_image_as_it_was() -> None:
    image = np.full((48, 64, 3), 7, dtype=np.uint8)
    assert (draw_analysis(image, FrameAnalysis(timestamp_s=0.0)) == image).all()
