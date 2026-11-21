from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from jetank.domain.perception import FloorMask, FrameAnalysis, NetworkOutputs


def test_outputs_are_found_by_their_rank() -> None:
    outputs = NetworkOutputs(
        tensors={
            "output0": np.zeros((1, 38, 10), np.float32),
            "output1": np.zeros((1, 32, 4, 4), np.float32),
        }
    )
    assert outputs.with_rank(4).shape == (1, 32, 4, 4)
    assert outputs.has_rank(3)
    assert not outputs.has_rank(2)
    with pytest.raises(ValueError, match="one output with 2 dimensions"):
        outputs.with_rank(2)


def test_outputs_need_a_tensor() -> None:
    with pytest.raises(ValidationError):
        NetworkOutputs(tensors={})


def test_floor_mask_is_a_bool_grid() -> None:
    cells = np.zeros((4, 8), dtype=bool)
    cells[2:] = True
    assert FloorMask(cells=cells).fraction == 0.5
    with pytest.raises(ValidationError):
        FloorMask(cells=np.zeros((4, 8), dtype=np.uint8))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FloorMask(cells=np.zeros((0, 8), dtype=bool))


def test_an_analysis_defaults_to_nothing_found() -> None:
    analysis = FrameAnalysis(timestamp_s=1.0)
    assert analysis.detections == ()
    assert analysis.floor is None
