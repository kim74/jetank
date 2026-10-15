"""
CSI camera driver using OpenCV with the NVIDIA Argus GStreamer source.

Layer: drivers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from jetank.domain.models import Frame
from jetank.errors import CameraError, DriverClosedError

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from jetank.config.settings import CameraConfig
    from jetank.interfaces.clock import Clock

logger = logging.getLogger(__name__)


class VideoCaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` this driver uses."""

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV naming

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


def build_pipeline(config: CameraConfig) -> str:
    """GStreamer pipeline: Argus capture in NVMM memory, hardware scale/flip, BGR to appsink.

    ``drop=true max-buffers=1`` keeps only the newest frame, so a slow consumer gets fresh
    images instead of a growing backlog.
    """
    return (
        f"nvarguscamerasrc sensor-id={config.sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){config.capture_width}, "
        f"height=(int){config.capture_height}, framerate=(fraction){config.fps}/1 ! "
        f"nvvidconv flip-method={config.flip_method} ! "
        f"video/x-raw, width=(int){config.width}, height=(int){config.height}, "
        "format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _opencv_capture(pipeline: str) -> VideoCaptureLike:
    import cv2

    capture: VideoCaptureLike = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return capture


class GstCamera:
    """:class:`~jetank.interfaces.Camera` for the IMX219 on a CSI port.

    Args:
        config: Camera settings.
        clock: Source of frame timestamps.
        capture_factory: Creates the capture object from a pipeline string; injectable
            so the driver can be tested without a camera.

    Raises:
        CameraError: The pipeline could not be opened.
    """

    def __init__(
        self,
        config: CameraConfig,
        clock: Clock,
        capture_factory: Callable[[str], VideoCaptureLike] = _opencv_capture,
    ) -> None:
        self._config = config
        self._clock = clock
        pipeline = build_pipeline(config)
        logger.debug("opening camera pipeline: %s", pipeline)
        self._capture = capture_factory(pipeline)
        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(
                f"could not open CSI camera sensor-id={config.sensor_id}; "
                "is nvargus-daemon running and the camera connected?"
            )
        self._closed = False

    def __enter__(self) -> GstCamera:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def read(self) -> Frame:
        if self._closed:
            raise DriverClosedError("camera is closed")
        ok, image = self._capture.read()
        if not ok or image is None:
            raise CameraError("camera returned no frame")
        expected = (self._config.height, self._config.width, 3)
        if not isinstance(image, np.ndarray) or image.shape != expected:
            shape = getattr(image, "shape", None)
            raise CameraError(f"unexpected frame shape {shape}, expected {expected}")
        return Frame(image=image, timestamp_s=self._clock.monotonic())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._capture.release()
