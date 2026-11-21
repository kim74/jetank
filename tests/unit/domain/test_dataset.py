from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.domain.dataset import (
    DatasetClass,
    DrawnImage,
    DrawnShape,
    Severity,
    Split,
    check_drawing,
    recording_runs,
    split_scenes,
    yolo_lines,
)
from jetank.domain.records import record_id_for
from jetank.domain.scenes import Lighting, SceneLabels, SceneRecord

T0 = datetime(2026, 11, 21, 17, 0, tzinfo=timezone.utc)
ID = record_id_for(T0)
FLOOR = DrawnShape(label="floor", kind="polygon", points=((0, 480), (0, 200), (640, 200)))
TOY = DrawnShape(label="toy_animal", kind="rectangle", points=((100, 300), (40, 250)))


def scene(seconds: float = 0, room: str = "kitchen", animals: bool = True) -> SceneRecord:
    moment = T0 + timedelta(seconds=seconds)
    labels = SceneLabels(room=room, lighting=Lighting.LAMP, has_animals=animals)
    return SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=labels)


def drawn(*shapes: DrawnShape, record_id: str = ID) -> DrawnImage:
    return DrawnImage(record_id=record_id, width_px=640, height_px=480, shapes=shapes)


def messages(image: DrawnImage, record: SceneRecord | None) -> list[tuple[Severity, str]]:
    return [(i.severity, i.message) for i in check_drawing(image, record)]


def test_class_indices_follow_the_declared_order() -> None:
    assert [c.class_index for c in DatasetClass] == [0, 1]
    assert DatasetClass.TOY_ANIMAL.class_index == 0


def test_a_consistent_drawing_has_no_issues() -> None:
    assert check_drawing(drawn(FLOOR, TOY), scene()) == []
    assert check_drawing(drawn(FLOOR), scene(animals=False)) == []


def test_animals_recorded_but_not_outlined_is_a_warning() -> None:
    assert messages(drawn(FLOOR), scene(animals=True)) == [
        (Severity.WARNING, "recorded with toy animals, but none is outlined")
    ]
    assert messages(drawn(FLOOR, TOY), scene(animals=False)) == [
        (Severity.WARNING, "toy animal outlined, but recorded without toy animals")
    ]


def test_no_floor_is_a_warning() -> None:
    assert messages(drawn(TOY), scene()) == [(Severity.WARNING, "no floor outlined")]


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (DrawnShape(label="cow", kind="polygon", points=FLOOR.points), "unknown class"),
        (DrawnShape(label="floor", kind="polygon", points=((0, 0), (1, 1))), "at least 3"),
        (DrawnShape(label="floor", kind="rectangle", points=((0, 0),)), "2 or 4 points"),
        (DrawnShape(label="floor", kind="circle", points=((0, 0), (5, 5))), "not supported"),
    ],
)
def test_unusable_shapes_are_errors(shape: DrawnShape, expected: str) -> None:
    errors = [
        i for i in check_drawing(drawn(FLOOR, shape), scene()) if i.severity is Severity.ERROR
    ]
    assert len(errors) == 1
    assert expected in errors[0].message
    assert "shape 2" in errors[0].message


def test_a_drawing_without_a_scene_is_an_error() -> None:
    issues = check_drawing(drawn(FLOOR), None)
    assert [i.severity for i in issues] == [Severity.ERROR]
    assert issues[0].record_id == ID


def test_yolo_lines_are_normalised_polygons() -> None:
    assert yolo_lines(drawn(FLOOR)) == ["1 0.000000 1.000000 0.000000 0.416667 1.000000 0.416667"]


def test_rectangles_become_four_corners_in_any_corner_order() -> None:
    two = yolo_lines(drawn(TOY))
    four = yolo_lines(
        drawn(
            DrawnShape(
                label="toy_animal",
                kind="rectangle",
                points=((40, 300), (100, 300), (100, 250), (40, 250)),
            )
        )
    )
    assert (
        two == four == ["0 0.062500 0.520833 0.156250 0.520833 0.156250 0.625000 0.062500 0.625000"]
    )


def test_points_on_or_past_the_edge_are_kept_inside() -> None:
    edge = DrawnShape(label="floor", kind="polygon", points=((-3, 490), (650, 490), (320, -1)))
    assert yolo_lines(drawn(edge)) == ["1 0.000000 1.000000 1.000000 1.000000 0.500000 0.000000"]


def test_an_empty_drawing_has_no_lines() -> None:
    assert yolo_lines(drawn()) == []


def test_yolo_lines_refuse_unchecked_shapes() -> None:
    with pytest.raises(ValueError, match="cow"):
        yolo_lines(drawn(DrawnShape(label="cow", kind="polygon", points=FLOOR.points)))
    with pytest.raises(ValueError, match="unusable circle"):
        yolo_lines(drawn(DrawnShape(label="floor", kind="circle", points=((0, 0), (1, 1)))))


def test_runs_break_at_pauses_and_room_changes() -> None:
    records = [scene(0), scene(2), scene(40), scene(42, room="hall"), scene(44, room="hall")]
    runs = recording_runs(reversed(records), gap_s=30)
    assert [[r.record_id for r in run] for run in runs] == [
        [records[0].record_id, records[1].record_id],
        [records[2].record_id],
        [records[3].record_id, records[4].record_id],
    ]


def test_split_sends_whole_blocks_to_val_in_every_run() -> None:
    kitchen = [scene(2 * i) for i in range(50)]  # 100 s: 5 blocks of 20 s, 10 scenes each
    hall = [scene(1000 + 2 * i, room="hall") for i in range(50)]
    splits = split_scenes([*kitchen, *hall], val_fraction=0.2, block_s=20)
    for run in (kitchen, hall):
        sides = [splits[r.record_id] for r in run]
        assert sides[20:30] == [Split.VAL] * 10  # the middle block
        assert sides.count(Split.VAL) == 10


def test_split_extremes_and_bad_arguments() -> None:
    records = [scene(2 * i) for i in range(30)]
    assert set(split_scenes(records, val_fraction=0.0).values()) == {Split.TRAIN}
    assert set(split_scenes(records, val_fraction=1.0).values()) == {Split.VAL}
    assert split_scenes([]) == {}
    with pytest.raises(ValueError, match="val_fraction"):
        split_scenes(records, val_fraction=1.5)
    with pytest.raises(ValueError, match="block_s"):
        split_scenes(records, block_s=0)


@given(
    gaps=st.lists(st.sampled_from([1.0, 2.0, 7.0, 100.0]), min_size=1, max_size=120),
    fraction=st.floats(min_value=0.0, max_value=0.5),
    block_s=st.sampled_from([5.0, 10.0, 20.0]),
)
def test_split_properties(gaps: list[float], fraction: float, block_s: float) -> None:
    times = [sum(gaps[:i]) for i in range(len(gaps))]
    records = [scene(t) for t in times]
    splits = split_scenes(records, val_fraction=fraction, block_s=block_s, run_gap_s=30)
    # every scene gets a side, and the same scenes always split the same way
    assert set(splits) == {r.record_id for r in records}
    assert splits == split_scenes(list(reversed(records)), fraction, block_s, 30)
    for run in recording_runs(records, 30):
        start = run[0].recorded_at
        blocks: dict[int, set[Split]] = {}
        for r in run:
            index = int((r.recorded_at - start).total_seconds() // block_s)
            blocks.setdefault(index, set()).add(splits[r.record_id])
        # scenes recorded within one block are on the same side
        assert all(len(sides) == 1 for sides in blocks.values())
        # each run's share of val blocks is the fraction, rounded to a whole block
        count = max(blocks) + 1
        val_blocks = sum(_is_val(i, fraction) for i in range(count))
        assert abs(val_blocks - count * fraction) <= 1
        assert all(sides == {Split.VAL} for i, sides in blocks.items() if _is_val(i, fraction))


def _is_val(block: int, fraction: float) -> bool:
    return math.floor((block + 1) * fraction + 0.5) > math.floor(block * fraction + 0.5)
