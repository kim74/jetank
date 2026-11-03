from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jetank.domain.coco import COCO_CLASSES
from jetank.domain.detection import BoundingBox, Detection, IntruderEvent, Response, Track

unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def boxes(draw: st.DrawFn) -> BoundingBox:
    x1, x2 = sorted((draw(unit), draw(unit)))
    y1, y2 = sorted((draw(unit), draw(unit)))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class TestBoundingBox:
    def test_whole_frame_is_accepted(self) -> None:
        assert box(0.0, 0.0, 1.0, 1.0).area == 1.0

    @pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
    def test_coordinates_outside_frame_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            box(value, 0.0, 1.0, 1.0)

    @pytest.mark.parametrize(
        "corners", [(0.6, 0.1, 0.5, 0.2), (0.1, 0.6, 0.2, 0.5)], ids=["x", "y"]
    )
    def test_inverted_corners_rejected(self, corners: tuple[float, float, float, float]) -> None:
        with pytest.raises(ValidationError, match="x1 <= x2"):
            box(*corners)

    def test_zero_size_box_is_allowed(self) -> None:
        assert box(0.5, 0.5, 0.5, 0.5).area == 0.0

    def test_geometry(self) -> None:
        b = box(0.1, 0.2, 0.5, 0.8)
        assert b.width == pytest.approx(0.4)
        assert b.height == pytest.approx(0.6)
        assert b.area == pytest.approx(0.24)
        assert b.center == pytest.approx((0.3, 0.5))

    def test_is_frozen(self) -> None:
        b = box(0.1, 0.1, 0.2, 0.2)
        with pytest.raises(ValidationError):
            b.x1 = 0.0  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox.model_validate({"x1": 0, "y1": 0, "x2": 1, "y2": 1, "z": 0})


class TestIou:
    def test_identical_boxes_overlap_fully(self) -> None:
        b = box(0.1, 0.1, 0.4, 0.4)
        assert b.iou(b) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self) -> None:
        assert box(0.0, 0.0, 0.2, 0.2).iou(box(0.5, 0.5, 0.7, 0.7)) == 0.0

    def test_boxes_sharing_an_edge_do_not_overlap(self) -> None:
        assert box(0.0, 0.0, 0.5, 0.5).iou(box(0.5, 0.0, 1.0, 0.5)) == 0.0

    def test_two_empty_boxes_give_zero_not_division_error(self) -> None:
        empty = box(0.3, 0.3, 0.3, 0.3)
        assert empty.iou(empty) == 0.0

    def test_half_shifted_boxes(self) -> None:
        # Two 0.2 x 0.2 boxes overlapping by half their width: 0.02 / (0.04 + 0.04 - 0.02).
        a = box(0.0, 0.0, 0.2, 0.2)
        b = box(0.1, 0.0, 0.3, 0.2)
        assert a.iou(b) == pytest.approx(1 / 3)

    def test_contained_box(self) -> None:
        outer = box(0.0, 0.0, 0.4, 0.4)
        inner = box(0.1, 0.1, 0.3, 0.3)
        assert outer.iou(inner) == pytest.approx(0.25)

    @given(boxes(), boxes())
    def test_symmetric_and_bounded(self, a: BoundingBox, b: BoundingBox) -> None:
        assert a.iou(b) == pytest.approx(b.iou(a))
        assert 0.0 <= a.iou(b) <= 1.0 + 1e-9

    @given(boxes())
    def test_self_overlap_is_one_unless_empty(self, b: BoundingBox) -> None:
        expected = 1.0 if b.area > 0.0 else 0.0
        assert b.iou(b) == pytest.approx(expected)


class TestDetection:
    def test_valid(self) -> None:
        d = Detection(label="cat", confidence=0.9, box=box(0.1, 0.1, 0.2, 0.2))
        assert d.label == "cat"

    def test_empty_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Detection(label="", confidence=0.9, box=box(0.1, 0.1, 0.2, 0.2))

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
    def test_confidence_outside_unit_range_rejected(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            Detection(label="cat", confidence=confidence, box=box(0.1, 0.1, 0.2, 0.2))


class TestTrack:
    def make(self, **overrides: object) -> Track:
        fields: dict[str, object] = {
            "track_id": 1,
            "label": "cat",
            "box": box(0.1, 0.1, 0.2, 0.2),
            "confidence": 0.8,
            "hits": 1,
            "first_seen_s": 10.0,
            "last_seen_s": 10.0,
        }
        fields.update(overrides)
        return Track.model_validate(fields)

    def test_new_track_is_unconfirmed(self) -> None:
        assert not self.make().confirmed

    @pytest.mark.parametrize(("field", "value"), [("track_id", 0), ("hits", 0)])
    def test_counts_start_at_one(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError, match=field):
            self.make(**{field: value})


def test_intruder_event_round_trips_through_json() -> None:
    event = IntruderEvent(
        track_id=3,
        label="cat",
        response=Response.CHASE,
        confidence=0.75,
        box=box(0.1, 0.2, 0.3, 0.4),
        timestamp_s=123.5,
    )
    assert IntruderEvent.model_validate_json(event.model_dump_json()) == event
    assert event.model_dump(mode="json")["response"] == "chase"


def test_response_parses_from_config_strings() -> None:
    assert [Response(v) for v in ("chase", "alert", "ignore")] == list(Response)


class TestCocoClasses:
    def test_has_the_80_coco_classes_without_duplicates(self) -> None:
        assert len(COCO_CLASSES) == 80
        assert len(set(COCO_CLASSES)) == 80

    @pytest.mark.parametrize(("index", "label"), [(0, "person"), (15, "cat"), (16, "dog")])
    def test_yolo_indices_of_the_classes_we_act_on(self, index: int, label: str) -> None:
        assert COCO_CLASSES[index] == label
