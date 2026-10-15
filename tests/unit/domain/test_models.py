from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from jetank.domain.models import DriveCommand, Frame, TrackSpeeds


class TestTrackSpeeds:
    def test_stopped_is_zero(self) -> None:
        speeds = TrackSpeeds.stopped()
        assert (speeds.left, speeds.right) == (0.0, 0.0)
        assert speeds.is_stopped

    @pytest.mark.parametrize("value", [-1.01, 1.01, math.nan, math.inf])
    def test_out_of_range_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            TrackSpeeds(left=value, right=0.0)

    def test_full_scale_accepted(self) -> None:
        speeds = TrackSpeeds(left=-1.0, right=1.0)
        assert not speeds.is_stopped

    def test_is_frozen(self) -> None:
        speeds = TrackSpeeds(left=0.1, right=0.2)
        with pytest.raises(ValidationError):
            speeds.left = 0.5  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrackSpeeds(left=0.0, right=0.0, middle=0.0)  # type: ignore[call-arg]

    def test_scaled_multiplies_both_sides(self) -> None:
        assert TrackSpeeds(left=0.8, right=-0.4).scaled(0.5) == TrackSpeeds(left=0.4, right=-0.2)

    @pytest.mark.parametrize("factor", [-0.1, 1.1])
    def test_scaled_rejects_factor_outside_unit_range(self, factor: float) -> None:
        with pytest.raises(ValueError, match="scale factor"):
            TrackSpeeds(left=0.5, right=0.5).scaled(factor)


class TestDriveCommand:
    def test_defaults_to_standing_still(self) -> None:
        assert DriveCommand() == DriveCommand(linear=0.0, angular=0.0)

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DriveCommand(linear=2.0)


class TestFrame:
    def test_valid_image_exposes_size(self) -> None:
        frame = Frame(image=np.zeros((48, 64, 3), dtype=np.uint8), timestamp_s=1.0)
        assert (frame.width, frame.height) == (64, 48)

    @pytest.mark.parametrize(
        "image",
        [
            np.zeros((48, 64), dtype=np.uint8),
            np.zeros((48, 64, 4), dtype=np.uint8),
            np.zeros((0, 64, 3), dtype=np.uint8),
            np.zeros((48, 64, 3), dtype=np.float32),
        ],
        ids=["grey", "bgra", "empty", "float"],
    )
    def test_invalid_image_rejected(self, image: np.ndarray) -> None:
        with pytest.raises(ValidationError):
            Frame(image=image, timestamp_s=0.0)

    def test_non_array_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Frame(image=[[0, 0, 0]], timestamp_s=0.0)  # type: ignore[arg-type]

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Frame(image=np.zeros((2, 2, 3), dtype=np.uint8), timestamp_s=-1.0)
