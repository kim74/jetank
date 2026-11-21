from __future__ import annotations

import numpy as np
import pytest

from jetank.perception.polygons import mask_polygons

pytest.importorskip("cv2")


def bounds(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def test_a_mask_reaching_the_edges_gives_an_outline_on_the_edges() -> None:
    cells = np.zeros((120, 160), dtype=bool)
    cells[60:] = True  # bottom half
    (polygon,) = mask_polygons(cells, 640, 480)
    assert bounds(polygon) == (0.0, pytest.approx(242.0), 640.0, 480.0)


def test_regions_come_largest_first_and_specks_are_dropped() -> None:
    cells = np.zeros((120, 160), dtype=bool)
    cells[100:, :40] = True  # 800 cells
    cells[10:40, 80:150] = True  # 2100 cells
    cells[5, 5] = True  # a speck
    polygons = mask_polygons(cells, 640, 480)
    assert len(polygons) == 2
    assert bounds(polygons[0])[0] > 300  # the larger, right-hand region first
    assert all(len(p) >= 3 for p in polygons)


def test_nothing_gives_no_outline() -> None:
    assert mask_polygons(np.zeros((120, 160), dtype=bool), 640, 480) == []
