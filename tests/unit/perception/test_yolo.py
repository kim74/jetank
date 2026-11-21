from __future__ import annotations

import numpy as np
import pytest

from jetank.domain.perception import NetworkOutputs
from jetank.perception.letterbox import Letterbox
from jetank.perception.yolo import YoloSettings, decode

from .yolo_outputs import CLASSES, PROTO, Candidate, outputs

# A 640 x 480 frame in a 640 x 640 input: no scaling, 80 px grey above and below.
BOX = Letterbox.fit(640, 480, 640, 640)
SETTINGS = YoloSettings(class_names=CLASSES, floor_label="floor")
TOY = (0.9, 0.05)
FLOOR = (0.05, 0.9)
# The whole input, with a mask in its bottom half.
WHOLE_FLOOR = Candidate(320, 320, 640, 640, FLOOR, coefficients=(1.0,))


def test_toys_are_reported_in_frame_coordinates() -> None:
    # 64 x 48 px box centred at (320, 320) in the input = (320, 240) in the frame
    result = decode(outputs(Candidate(320, 320, 64, 48, TOY)), BOX, SETTINGS, 12.5)
    assert result.timestamp_s == 12.5
    (toy,) = result.detections
    assert toy.label == "toy_animal"
    assert toy.confidence == pytest.approx(0.9)
    assert (toy.box.x1, toy.box.x2) == pytest.approx((0.45, 0.55))
    assert (toy.box.y1, toy.box.y2) == pytest.approx((0.45, 0.55))


def test_weak_candidates_are_dropped() -> None:
    result = decode(outputs(Candidate(320, 320, 64, 48, (0.39, 0.0))), BOX, SETTINGS, 0.0)
    assert result.detections == ()


def test_overlapping_boxes_of_one_class_are_merged_best_first() -> None:
    result = decode(
        outputs(
            Candidate(320, 320, 64, 64, (0.6, 0.0)),
            Candidate(322, 322, 64, 64, (0.8, 0.0)),  # same toy, better score
            Candidate(100, 300, 40, 40, (0.5, 0.0)),  # another toy
        ),
        BOX,
        SETTINGS,
        0.0,
    )
    assert [round(d.confidence, 2) for d in result.detections] == [0.8, 0.5]


def test_boxes_of_different_classes_do_not_suppress_each_other() -> None:
    settings = YoloSettings(class_names=("cat", "dog"))
    result = decode(
        outputs(
            Candidate(320, 320, 64, 64, (0.8, 0.0)),
            Candidate(320, 320, 64, 64, (0.0, 0.7)),
            masks=False,
        ),
        BOX,
        settings,
        0.0,
    )
    assert sorted(d.label for d in result.detections) == ["cat", "dog"]
    assert result.floor is None


def test_at_most_max_detections_are_kept() -> None:
    settings = SETTINGS.model_copy(update={"max_detections": 2})
    candidates = [Candidate(40 + 60 * i, 320, 30, 30, (0.5 + i / 100, 0.0)) for i in range(5)]
    result = decode(outputs(*candidates), BOX, settings, 0.0)
    assert [round(d.confidence, 2) for d in result.detections] == [0.54, 0.53]


def test_floor_becomes_a_mask_over_the_frame_not_a_detection() -> None:
    result = decode(outputs(WHOLE_FLOOR, Candidate(320, 500, 40, 40, TOY)), BOX, SETTINGS, 0.0)
    assert [d.label for d in result.detections] == ["toy_animal"]
    assert result.floor is not None
    cells = result.floor.cells
    # The frame covers prototype rows 20..140 of 160; the mask is the input's bottom half,
    # i.e. rows 80..140 of the prototypes = the frame's bottom half.
    assert cells.shape == (120, 160)
    assert not cells[:60].any()
    assert cells[60:].all()
    assert result.floor.fraction == pytest.approx(0.5)


def test_each_floor_mask_is_cut_to_its_box() -> None:
    # Box over the left half of the input only: the bottom-left quarter is floor.
    left = Candidate(160, 320, 320, 640, FLOOR, coefficients=(1.0,))
    result = decode(outputs(left), BOX, SETTINGS, 0.0)
    assert result.floor is not None
    cells = result.floor.cells
    assert cells[60:, : PROTO // 2].all()
    assert not cells[:, PROTO // 2 :].any()


def test_no_floor_found_means_no_floor_anywhere() -> None:
    result = decode(outputs(Candidate(320, 320, 64, 48, TOY)), BOX, SETTINGS, 0.0)
    assert result.floor is not None
    assert not result.floor.cells.any()


def test_a_detection_model_has_no_floor_mask() -> None:
    settings = YoloSettings(class_names=CLASSES)
    result = decode(outputs(Candidate(320, 320, 64, 48, TOY), masks=False), BOX, settings, 0.0)
    assert len(result.detections) == 1
    assert result.floor is None


@pytest.mark.parametrize(
    "tensors",
    [
        outputs(classes=3),  # model trained with 3 classes, config names 2
        outputs(masks=False, classes=1),
        NetworkOutputs(tensors={"a": np.zeros((2, 38, 10), np.float32)}),  # batch of 2
    ],
)
def test_outputs_that_do_not_fit_the_config_are_refused(tensors: NetworkOutputs) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        decode(tensors, BOX, SETTINGS, 0.0)


def test_floor_label_must_be_a_class() -> None:
    with pytest.raises(ValueError, match="floor_label"):
        YoloSettings(class_names=("cat",), floor_label="floor")


def test_odd_numbers_from_a_real_engine_do_not_break_the_domain_rules() -> None:
    result = decode(
        outputs(Candidate(320, 320, 64, 48, (1.0004, 0.0)), Candidate(100, 300, -10, 20, TOY)),
        BOX,
        SETTINGS,
        0.0,
    )
    assert [d.confidence for d in result.detections] == [1.0, pytest.approx(0.9)]
    assert result.detections[1].box.width == 0.0
