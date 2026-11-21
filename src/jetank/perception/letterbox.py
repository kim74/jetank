"""
Letterboxing: fitting a camera frame into the network's square input without distorting it.

Layer: perception

The frame is scaled to fit and centred, and the rest is filled with grey (114), exactly as
Ultralytics does when training, so the model sees images the way it learned them. The
:class:`Letterbox` remembers the scale and padding, to map what the network finds back
onto the frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Grey the padding is filled with (Ultralytics' value).
PAD_VALUE: Final = 114


class Letterbox(BaseModel):
    """How a frame was placed in the network input.

    Attributes:
        frame_width / frame_height: Size of the original frame in pixels.
        input_width / input_height: Size of the network input in pixels.
        scale: Input pixels per frame pixel.
        pad_x / pad_y: Padding left of and above the scaled frame, in input pixels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    input_height: int = Field(gt=0)
    scale: float = Field(gt=0.0)
    pad_x: float = Field(ge=0.0)
    pad_y: float = Field(ge=0.0)

    @classmethod
    def fit(
        cls, frame_width: int, frame_height: int, input_width: int, input_height: int
    ) -> Letterbox:
        """The placement of a frame of this size in an input of that size."""
        scale = min(input_width / frame_width, input_height / frame_height)
        scaled_w = round(frame_width * scale)
        scaled_h = round(frame_height * scale)
        return cls(
            frame_width=frame_width,
            frame_height=frame_height,
            input_width=input_width,
            input_height=input_height,
            scale=scale,
            pad_x=(input_width - scaled_w) / 2,
            pad_y=(input_height - scaled_h) / 2,
        )

    def to_frame(
        self, x: NDArray[np.float32], y: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Input pixel coordinates to normalised frame coordinates (0..1, clipped)."""
        fx = (x - self.pad_x) / self.scale / self.frame_width
        fy = (y - self.pad_y) / self.scale / self.frame_height
        return np.clip(fx, 0.0, 1.0), np.clip(fy, 0.0, 1.0)

    def frame_region(self, grid_width: int, grid_height: int) -> tuple[slice, slice]:
        """The rows and columns of a grid over the input (e.g. 160 x 160 mask prototypes)
        that cover the frame rather than the padding."""
        sx = grid_width / self.input_width
        sy = grid_height / self.input_height
        top = round(self.pad_y * sy)
        bottom = round((self.input_height - self.pad_y) * sy)
        left = round(self.pad_x * sx)
        right = round((self.input_width - self.pad_x) * sx)
        return slice(top, bottom), slice(left, right)


def letterbox(
    image_bgr: NDArray[np.uint8], input_width: int, input_height: int
) -> tuple[NDArray[np.float32], Letterbox]:
    """The network input for a BGR camera image.

    Args:
        image_bgr: ``uint8`` array ``(height, width, 3)`` in BGR order, as the camera gives it.
        input_width / input_height: The network's input size in pixels.

    Returns:
        A ``float32`` batch ``(1, 3, input_height, input_width)``: RGB, scaled to 0..1, and
        the placement, to map results back.
    """
    height, width = image_bgr.shape[:2]
    box = Letterbox.fit(width, height, input_width, input_height)
    scaled_w = round(width * box.scale)
    scaled_h = round(height * box.scale)
    if (scaled_w, scaled_h) != (width, height):
        import cv2  # only needed to resize; imported here like elsewhere in the package

        image_bgr = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), PAD_VALUE, dtype=np.uint8)
    top = round(box.pad_y - 0.1)  # Ultralytics rounds the padding this way
    left = round(box.pad_x - 0.1)
    canvas[top : top + scaled_h, left : left + scaled_w] = image_bgr
    rgb = canvas[:, :, ::-1]
    batch = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32) / 255.0
    return batch.astype(np.float32, copy=False), box
