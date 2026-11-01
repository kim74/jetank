from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedError, InvalidStatus

from jetank.app.teleop_server import TeleopServer, load_page
from jetank.app.video_stream import CLOSE_CAMERA_FAILED, VideoStream, encode_jpeg
from jetank.config.settings import DriveConfig, TeleopConfig
from jetank.control.drive import DriveController
from jetank.control.teleop import TeleopSession
from jetank.domain.models import Frame
from jetank.drivers.fake import FakeCamera, FakeMotorDriver
from jetank.drivers.system_clock import SystemClock
from jetank.errors import CameraError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

CONFIG = TeleopConfig(host="127.0.0.1", port=0, tick_period_s=0.01)
FPS = 30.0


def frame_number(frame: Frame) -> bytes:
    """Test encoder: the fake camera writes its frame counter into the red channel."""
    return bytes([int(frame.image[0, 0, 2])])


class FailingCamera(FakeCamera):
    """Delivers ``good_frames`` frames, then fails like a camera whose cable came loose."""

    def __init__(self, good_frames: int) -> None:
        super().__init__(8, 6, SystemClock())
        self._good_frames = good_frames

    def read(self) -> Frame:
        if self.frames_read >= self._good_frames:
            raise CameraError("camera returned no frame")
        return super().read()


class BlockingCamera(FakeCamera):
    """A camera whose read never returns until released, like a hung pipeline."""

    def __init__(self) -> None:
        super().__init__(8, 6, SystemClock())
        self.reading = threading.Event()
        self.release = threading.Event()

    def read(self) -> Frame:
        self.reading.set()
        self.release.wait(5)
        return super().read()


async def drain(viewer: ClientConnection) -> None:
    """Receive until the server closes the connection."""
    while True:
        await asyncio.wait_for(viewer.recv(), 5)


def run_with_video(video: VideoStream | None, body: Callable[[int], Awaitable[None]]) -> None:
    """Start a teleop server with ``video`` on a free local port, run ``body(port)``, stop it."""

    async def main() -> None:
        drive = DriveController(FakeMotorDriver(), SystemClock(), DriveConfig())
        server = TeleopServer(TeleopSession(drive), CONFIG, b"page", video)
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(server.serve(on_ready=ready.set_result))
        port = await asyncio.wait_for(ready, 5)
        try:
            await body(port)
        finally:
            server.request_stop()
            await asyncio.wait_for(task, 5)

    asyncio.run(main())


def test_viewer_receives_consecutive_frames() -> None:
    camera = FakeCamera(8, 6, SystemClock())
    video = VideoStream(camera, FPS, frame_number)

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/video") as viewer:
            frames = [await asyncio.wait_for(viewer.recv(), 5) for _ in range(3)]
        assert all(isinstance(f, bytes) and len(f) == 1 for f in frames)
        numbers = [f[0] for f in frames]
        assert numbers == sorted(set(numbers)), "each message is a newer frame"

    run_with_video(video, body)
    video.close()


def test_two_viewers_share_one_camera() -> None:
    camera = FakeCamera(8, 6, SystemClock())
    video = VideoStream(camera, FPS, frame_number)

    async def body(port: int) -> None:
        url = f"ws://127.0.0.1:{port}/video"
        async with connect(url) as first, connect(url) as second:
            for _ in range(3):
                await asyncio.wait_for(first.recv(), 5)
                await asyncio.wait_for(second.recv(), 5)
        # Shared capture: far fewer reads than if each viewer had its own loop.
        assert camera.frames_read < 6

    run_with_video(video, body)
    video.close()


def test_camera_is_idle_without_viewers() -> None:
    camera = FakeCamera(8, 6, SystemClock())
    video = VideoStream(camera, FPS, frame_number)

    async def body(port: int) -> None:
        await asyncio.sleep(0.2)
        assert camera.frames_read == 0
        async with connect(f"ws://127.0.0.1:{port}/video") as viewer:
            await asyncio.wait_for(viewer.recv(), 5)
        await asyncio.sleep(0.1)  # let the loop notice the viewer left
        reads_after_leaving = camera.frames_read
        await asyncio.sleep(0.2)
        assert camera.frames_read <= reads_after_leaving + 1

    run_with_video(video, body)
    video.close()


def test_frame_rate_is_limited() -> None:
    camera = FakeCamera(8, 6, SystemClock())
    video = VideoStream(camera, 10.0, frame_number)

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/video") as viewer:
            await asyncio.wait_for(viewer.recv(), 5)
            await asyncio.sleep(0.5)
        assert camera.frames_read <= 8  # ~6 at 10 fps; an unthrottled loop reads hundreds

    run_with_video(video, body)
    video.close()


def test_camera_failure_closes_viewers_but_driving_continues() -> None:
    video = VideoStream(FailingCamera(good_frames=2), FPS, frame_number)

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/video") as viewer:
            with pytest.raises(ConnectionClosedError) as closed:
                await drain(viewer)
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.code == CLOSE_CAMERA_FAILED
        assert "camera failed" in closed.value.rcvd.reason
        assert video.failed
        with pytest.raises(InvalidStatus, match="503"):
            async with connect(f"ws://127.0.0.1:{port}/video"):
                pass
        async with connect(f"ws://127.0.0.1:{port}/ws") as driver:
            await asyncio.wait_for(driver.recv(), 5)
            await driver.send(json.dumps({"type": "drive", "linear": 0.5, "angular": 0.0}))
            assert json.loads(await asyncio.wait_for(driver.recv(), 5))["moving"] is True

    run_with_video(video, body)
    video.close()


def test_unexpected_encoder_error_ends_video_cleanly() -> None:
    def broken_encoder(_frame: Frame) -> bytes:
        raise RuntimeError("cv2.error stand-in")

    video = VideoStream(FakeCamera(8, 6, SystemClock()), FPS, broken_encoder)

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/video") as viewer:
            with pytest.raises(ConnectionClosedError) as closed:
                await drain(viewer)
        assert closed.value.rcvd is not None
        assert closed.value.rcvd.reason == "video error: RuntimeError"

    run_with_video(video, body)  # shuts down without re-raising
    video.close()


def test_no_camera_means_video_404() -> None:
    async def body(port: int) -> None:
        with pytest.raises(InvalidStatus, match="404"):
            async with connect(f"ws://127.0.0.1:{port}/video"):
                pass

    run_with_video(None, body)


def test_shutdown_does_not_wait_for_a_hung_camera_read() -> None:
    camera = BlockingCamera()
    video = VideoStream(camera, FPS, frame_number)

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/video"):
            assert await asyncio.to_thread(camera.reading.wait, 5)

    try:
        run_with_video(video, body)  # returns although the read is still blocked
        assert not camera.release.is_set()
    finally:
        video.close()
        camera.release.set()


def test_fps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoStream(FakeCamera(8, 6, SystemClock()), 0.0, frame_number)


def test_encode_jpeg_produces_a_jpeg() -> None:
    pytest.importorskip("cv2")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    data = encode_jpeg(Frame(image=image, timestamp_s=0.0), quality=70)
    assert data[:2] == b"\xff\xd8"  # JPEG start-of-image marker


def test_packaged_page_shows_live_video() -> None:
    page = load_page().decode()
    assert 'id="video"' in page
    assert "/video" in page
    assert "VIDEO_STALL_MS" in page
