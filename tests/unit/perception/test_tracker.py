from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.config.settings import TrackerConfig
from jetank.domain.detection import BoundingBox, Detection
from jetank.perception.tracker import IouTracker

CONFIG = TrackerConfig(match_iou_threshold=0.3, confirm_hits=3, max_age_s=1.0)


def toy(x: float, label: str = "toy_animal", size: float = 0.1) -> Detection:
    return Detection(
        label=label, confidence=0.8, box=BoundingBox(x1=x, y1=0.5, x2=x + size, y2=0.5 + size)
    )


def test_a_toy_seen_in_successive_frames_keeps_its_track_and_is_confirmed() -> None:
    tracker = IouTracker(CONFIG)
    (first,) = tracker.update([toy(0.40)], 0.0)
    assert (first.track_id, first.hits, first.confirmed) == (1, 1, False)
    tracker.update([toy(0.41)], 0.1)
    (third,) = tracker.update([toy(0.42)], 0.2)
    assert (third.track_id, third.hits, third.confirmed) == (1, 3, True)
    assert third.box == toy(0.42).box
    assert (third.first_seen_s, third.last_seen_s) == (0.0, 0.2)


def test_separate_toys_get_separate_tracks() -> None:
    tracker = IouTracker(CONFIG)
    tracks = tracker.update([toy(0.1), toy(0.6)], 0.0)
    assert [t.track_id for t in tracks] == [1, 2]
    tracks = tracker.update([toy(0.61), toy(0.11)], 0.1)
    assert {t.track_id: round(t.box.x1, 2) for t in tracks} == {1: 0.11, 2: 0.61}


def test_labels_never_match_each_other() -> None:
    tracker = IouTracker(CONFIG)
    tracker.update([toy(0.4, "cat")], 0.0)
    tracks = tracker.update([toy(0.4, "dog")], 0.1)
    assert [(t.track_id, t.label) for t in tracks] == [(1, "cat"), (2, "dog")]


def test_a_detection_far_from_every_track_starts_a_new_one() -> None:
    tracker = IouTracker(CONFIG)
    tracker.update([toy(0.1)], 0.0)
    tracks = tracker.update([toy(0.5)], 0.1)
    assert [t.track_id for t in tracks] == [1, 2]


def test_the_best_overlap_wins_when_two_detections_compete() -> None:
    tracker = IouTracker(CONFIG)
    tracker.update([toy(0.40)], 0.0)
    tracks = tracker.update([toy(0.44), toy(0.41)], 0.1)
    by_id = {t.track_id: round(t.box.x1, 2) for t in tracks}
    assert by_id == {1: 0.41, 2: 0.44}


def test_tracks_unseen_for_longer_than_max_age_are_dropped() -> None:
    tracker = IouTracker(CONFIG)
    tracker.update([toy(0.4)], 0.0)
    assert [t.track_id for t in tracker.update([], 1.0)] == [1]  # still within max_age
    assert tracker.update([], 1.01) == []
    assert [t.track_id for t in tracker.update([toy(0.4)], 1.1)] == [2]  # ids are never reused


def test_frames_must_come_in_order_and_reset_forgets() -> None:
    tracker = IouTracker(CONFIG)
    tracker.update([toy(0.4)], 1.0)
    with pytest.raises(ValueError, match="capture order"):
        tracker.update([toy(0.4)], 0.5)
    tracker.reset()
    assert tracker.tracks == []
    assert tracker.update([toy(0.4)], 0.5)[0].track_id == 2


def test_one_hit_confirms_when_configured_so() -> None:
    tracker = IouTracker(CONFIG.model_copy(update={"confirm_hits": 1}))
    assert tracker.update([toy(0.4)], 0.0)[0].confirmed


@given(
    st.lists(st.lists(st.floats(min_value=0.0, max_value=0.9), max_size=4), min_size=1, max_size=15)
)
def test_each_detection_continues_or_starts_exactly_one_track(frames: list[list[float]]) -> None:
    tracker = IouTracker(CONFIG)
    for index, xs in enumerate(frames):
        now = index * 0.1
        tracks = tracker.update([toy(x) for x in xs], now)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))
        assert sum(t.last_seen_s == now for t in tracks) == len(xs)
