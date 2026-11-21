from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.perception.letterbox import PAD_VALUE, Letterbox, letterbox


def test_a_camera_frame_is_centred_in_a_square_input() -> None:
    box = Letterbox.fit(640, 480, 640, 640)
    assert (box.scale, box.pad_x, box.pad_y) == (1.0, 0.0, 80.0)
    assert box.frame_region(160, 160) == (slice(20, 140), slice(0, 160))


def test_the_input_is_rgb_scaled_to_one_with_grey_padding() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    batch, box = letterbox(image, 640, 640)
    assert batch.shape == (1, 3, 640, 640)
    assert batch.dtype == np.float32
    assert box.pad_y == 80
    assert batch[0, :, 79, 0] == pytest.approx([PAD_VALUE / 255] * 3)  # padding
    assert batch[0, :, 80, 0] == pytest.approx([0.0, 0.0, 1.0])  # blue is the third channel


def test_larger_frames_are_scaled_down() -> None:
    pytest.importorskip("cv2")
    image = np.full((960, 1280, 3), 200, dtype=np.uint8)
    batch, box = letterbox(image, 640, 640)
    assert box.scale == 0.5
    assert batch[0, 0, 320, 320] == pytest.approx(200 / 255)


@given(
    width=st.integers(min_value=16, max_value=2000),
    height=st.integers(min_value=16, max_value=2000),
    u=st.floats(min_value=0.0, max_value=1.0),
    v=st.floats(min_value=0.0, max_value=1.0),
)
def test_frame_points_map_back_to_themselves(width: int, height: int, u: float, v: float) -> None:
    box = Letterbox.fit(width, height, 640, 640)
    x = np.array([box.pad_x + u * width * box.scale], dtype=np.float32)
    y = np.array([box.pad_y + v * height * box.scale], dtype=np.float32)
    fx, fy = box.to_frame(x, y)
    assert fx[0] == pytest.approx(u, abs=1e-4)
    assert fy[0] == pytest.approx(v, abs=1e-4)


def test_points_in_the_padding_are_clipped_onto_the_frame() -> None:
    box = Letterbox.fit(640, 480, 640, 640)
    fx, fy = box.to_frame(np.array([-5.0, 700.0]), np.array([10.0, 630.0]))
    assert list(fx) == [0.0, 1.0]
    assert list(fy) == [0.0, 1.0]
