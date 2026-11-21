"""
Outlines of a mask: the floor grid as polygons a labelling tool can edit.

Layer: perception
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Regions smaller than this share of the frame are left out (specks, not floor).
DEFAULT_MIN_AREA_FRACTION: Final = 0.01
#: How far a simplified outline may stray from the mask, in mask cells.
DEFAULT_TOLERANCE_CELLS: Final = 1.0


def mask_polygons(
    cells: NDArray[np.bool_],
    width_px: int,
    height_px: int,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    tolerance_cells: float = DEFAULT_TOLERANCE_CELLS,
) -> list[list[tuple[float, float]]]:
    """The outer outline of each region of ``cells``, largest first, in image pixels.

    Holes inside a region are not cut out (a polygon cannot have holes).

    Args:
        cells: ``bool`` grid over the image, ``True`` inside.
        width_px / height_px: Size of the image the outlines are for.
        min_area_fraction: Smaller regions are dropped.
        tolerance_cells: Simplification: points closer than this to the outline are removed.

    Returns:
        Polygons as ``(x, y)`` points (at least 3 each).
    """
    import cv2  # imported here like elsewhere in the package

    rows, cols = cells.shape
    contours, _ = cv2.findContours(
        cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sx, sy = width_px / cols, height_px / rows
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area_fraction * rows * cols:
            continue
        simple = cv2.approxPolyDP(contour, tolerance_cells, closed=True).reshape(-1, 2)
        if len(simple) < 3:
            continue
        polygons.append(
            [
                (_to_px(int(c), cols, sx, width_px), _to_px(int(r), rows, sy, height_px))
                for c, r in simple
            ]
        )
    return polygons


def _to_px(cell: int, cells: int, scale: float, size_px: int) -> float:
    """A contour point on cell ``cell`` in pixels: the cell's centre, or the image border for
    the outermost cells (so a mask reaching the edge yields an outline on the edge)."""
    if cell <= 0:
        return 0.0
    if cell >= cells - 1:
        return float(size_px)
    return (cell + 0.5) * scale
