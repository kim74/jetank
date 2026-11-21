"""
What the camera model sees in one frame: the objects found and where the floor is.

Layer: domain

The floor is a coarse grid of booleans over the whole frame (row 0 at the top), at the
model's mask resolution (e.g. 120 x 160 for a 640 x 480 frame). It is what the free-space
scan reads later: in each column, the first non-floor cell from the bottom is the nearest
obstacle in that direction.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jetank.domain.detection import Detection

_ARRAYS: Final = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class NetworkOutputs(BaseModel):
    """The output tensors of one network run, by the network's output name.

    Attributes:
        tensors: ``float32`` arrays, e.g. ``{"output0": (1, 38, 8400), "output1": (1, 32,
            160, 160)}`` for a YOLO segmentation model.
    """

    model_config = _ARRAYS

    tensors: dict[str, NDArray[np.float32]] = Field(min_length=1)

    def with_rank(self, rank: int) -> NDArray[np.float32]:
        """The one output with ``rank`` dimensions.

        Raises:
            ValueError: No output, or more than one, has that many dimensions.
        """
        found = [t for t in self.tensors.values() if t.ndim == rank]
        if len(found) != 1:
            shapes = [t.shape for t in self.tensors.values()]
            raise ValueError(f"expected one output with {rank} dimensions, got shapes {shapes}")
        return found[0]

    def has_rank(self, rank: int) -> bool:
        """Whether some output has ``rank`` dimensions."""
        return any(t.ndim == rank for t in self.tensors.values())


class FloorMask(BaseModel):
    """Which parts of the frame show drivable floor.

    Attributes:
        cells: ``bool`` array ``(rows, columns)`` covering the frame; ``True`` = floor.
    """

    model_config = _ARRAYS

    cells: NDArray[np.bool_]

    @field_validator("cells")
    @classmethod
    def _check_cells(cls, value: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
            raise ValueError("floor cells must be a bool numpy array")
        if value.ndim != 2 or 0 in value.shape:
            raise ValueError(f"floor cells must be a non-empty 2-D array, got {value.shape}")
        return value

    @property
    def fraction(self) -> float:
        """Share of the frame that is floor, 0..1."""
        return float(self.cells.mean())


class FrameAnalysis(BaseModel):
    """Everything the model found in one frame.

    Attributes:
        timestamp_s: Capture time of the frame (monotonic seconds), so stale results show.
        detections: Objects found, best first; the floor is not among them.
        floor: Where the floor is, or ``None`` if the model does not segment the floor.
    """

    model_config = _ARRAYS

    timestamp_s: float = Field(ge=0.0)
    detections: tuple[Detection, ...] = ()
    floor: FloorMask | None = None
