"""Real IMX219 camera on CAM0. Run with `pytest -m hardware`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from jetank.config.settings import CameraConfig
from jetank.drivers.gst_camera import GstCamera
from jetank.drivers.system_clock import SystemClock

pytestmark = pytest.mark.hardware


def test_csi_camera_delivers_frames() -> None:
    config = CameraConfig(backend="hardware")
    with GstCamera(config, SystemClock()) as camera:
        frames = [camera.read() for _ in range(15)]
    last = frames[-1]
    assert last.image.shape == (config.height, config.width, 3)
    assert all(b.timestamp_s >= a.timestamp_s for a, b in itertools.pairwise(frames))
    # A working sensor yields some contrast once auto-exposure has settled.
    assert float(np.std(last.image)) > 1.0
