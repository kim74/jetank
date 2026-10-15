"""
Synthetic camera producing deterministic test images.

Layer: drivers (fake)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jetank.domain.models import Frame
from jetank.errors import DriverClosedError

if TYPE_CHECKING:
    from jetank.interfaces.clock import Clock


class FakeCamera:
    """:class:`~jetank.interfaces.Camera` returning a gradient whose brightness encodes the
    frame number, so tests can tell consecutive frames apart."""

    def __init__(self, width: int, height: int, clock: Clock) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self._width = width
        self._height = height
        self._clock = clock
        self.frames_read = 0
        self.closed = False

    def read(self) -> Frame:
        if self.closed:
            raise DriverClosedError("camera is closed")
        ramp = np.linspace(0, 255, self._width, dtype=np.uint8)
        image = np.broadcast_to(ramp[np.newaxis, :, np.newaxis], (self._height, self._width, 3))
        image = image.copy()
        image[..., 2] = self.frames_read % 256
        self.frames_read += 1
        return Frame(image=image, timestamp_s=self._clock.monotonic())

    def close(self) -> None:
        self.closed = True
