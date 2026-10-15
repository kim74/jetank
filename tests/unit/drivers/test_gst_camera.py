from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from jetank.config.settings import CameraConfig
from jetank.drivers.gst_camera import GstCamera, build_pipeline
from jetank.errors import CameraError, DriverClosedError
from jetank.interfaces import Camera

if TYPE_CHECKING:
    from jetank.drivers.fake import FakeClock


class StubCapture:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, *, opened: bool = True, frames: list[Any] | None = None) -> None:
        self.opened = opened
        self.frames = list(frames or [])
        self.release_count = 0
        self.pipeline: str | None = None

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def read(self) -> tuple[bool, Any]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.release_count += 1


def make_camera(config: CameraConfig, clock: FakeClock, capture: StubCapture) -> GstCamera:
    def factory(pipeline: str) -> StubCapture:
        capture.pipeline = pipeline
        return capture

    return GstCamera(config, clock, capture_factory=factory)


def blank(config: CameraConfig) -> np.ndarray:
    return np.zeros((config.height, config.width, 3), dtype=np.uint8)


def test_pipeline_reflects_config() -> None:
    config = CameraConfig(
        sensor_id=1,
        capture_width=1280,
        capture_height=720,
        fps=60,
        width=320,
        height=240,
        flip_method=2,
    )
    pipeline = build_pipeline(config)
    assert pipeline.startswith("nvarguscamerasrc sensor-id=1 ")
    assert "width=(int)1280, height=(int)720, framerate=(fraction)60/1" in pipeline
    assert "nvvidconv flip-method=2" in pipeline
    assert "width=(int)320, height=(int)240" in pipeline
    assert pipeline.endswith("appsink drop=true max-buffers=1 sync=false")


def test_read_returns_timestamped_frame(camera_config: CameraConfig, clock: FakeClock) -> None:
    capture = StubCapture(frames=[blank(camera_config)])
    camera = make_camera(camera_config, clock, capture)
    frame = camera.read()
    assert isinstance(camera, Camera)
    assert capture.pipeline == build_pipeline(camera_config)
    assert (frame.width, frame.height) == (camera_config.width, camera_config.height)
    assert frame.timestamp_s == clock.monotonic()


def test_unopened_pipeline_raises_and_releases(
    camera_config: CameraConfig, clock: FakeClock
) -> None:
    capture = StubCapture(opened=False)
    with pytest.raises(CameraError, match="sensor-id=0"):
        make_camera(camera_config, clock, capture)
    assert capture.release_count == 1


def test_missing_frame_raises(camera_config: CameraConfig, clock: FakeClock) -> None:
    camera = make_camera(camera_config, clock, StubCapture(frames=[]))
    with pytest.raises(CameraError, match="no frame"):
        camera.read()


def test_wrong_frame_shape_raises(camera_config: CameraConfig, clock: FakeClock) -> None:
    wrong = np.zeros((10, 10, 3), dtype=np.uint8)
    camera = make_camera(camera_config, clock, StubCapture(frames=[wrong]))
    with pytest.raises(CameraError, match="unexpected frame shape"):
        camera.read()


def test_close_releases_once_and_blocks_reads(
    camera_config: CameraConfig, clock: FakeClock
) -> None:
    capture = StubCapture(frames=[blank(camera_config)])
    with make_camera(camera_config, clock, capture) as camera:
        pass
    camera.close()
    assert capture.release_count == 1
    with pytest.raises(DriverClosedError):
        camera.read()
