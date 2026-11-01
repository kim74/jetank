"""
Live camera video for the remote-control page: JPEG frames pushed over a WebSocket.

Layer: app

One capture loop reads and encodes frames in its own worker thread, so a slow camera never
blocks the event loop that runs the drive watchdog. The thread is deliberately not the event
loop's default executor: ``asyncio.run`` waits for that one on shutdown, so a hung camera read
would delay stopping the motors. It only captures while at least one viewer is
connected. Every viewer always gets the newest frame: a slow client skips frames instead of
building up a backlog, so the picture it shows is never old.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from websockets.exceptions import ConnectionClosed

from jetank.errors import CameraError, JetankError

if TYPE_CHECKING:
    from collections.abc import Callable

    from websockets.asyncio.server import ServerConnection

    from jetank.domain.models import Frame
    from jetank.interfaces.camera import Camera

logger = logging.getLogger(__name__)

#: WebSocket close code "internal error", sent to viewers when the camera fails.
CLOSE_CAMERA_FAILED = 1011


def encode_jpeg(frame: Frame, quality: int) -> bytes:
    """Compress a BGR frame to JPEG.

    Args:
        frame: Image to encode.
        quality: JPEG quality, 1-100.

    Raises:
        CameraError: OpenCV could not encode the image.
    """
    import cv2

    ok, data = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraError("could not encode the frame as JPEG")
    return bytes(data.tobytes())


class VideoStream:
    """Shares one camera between any number of WebSocket viewers.

    Args:
        camera: Frame source; owned (and closed) by the caller.
        fps: Upper limit on frames captured and sent per second.
        encode: Turns a frame into the bytes sent to viewers (normally JPEG).
    """

    def __init__(self, camera: Camera, fps: float, encode: Callable[[Frame], bytes]) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._camera = camera
        self._period_s = 1.0 / fps
        self._encode = encode
        self._viewers = 0
        self._has_viewers = asyncio.Event()
        # Replaced by a fresh event after each change, so waiting viewers wake exactly once.
        self._changed = asyncio.Event()
        self._latest: bytes | None = None
        self._sequence = 0
        self._failure: str | None = None
        self._stopped = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")

    @property
    def failed(self) -> bool:
        """True once the camera has failed; the stream does not recover."""
        return self._failure is not None

    async def run(self) -> None:
        """Capture frames while anyone watches, until cancelled or the camera fails.

        A camera failure is logged and ends the stream (viewers are disconnected with the
        reason); it is not raised, because driving can continue without video.
        """
        loop = asyncio.get_running_loop()
        try:
            while True:
                await self._has_viewers.wait()
                started = loop.time()
                data = await loop.run_in_executor(self._executor, self._capture)
                self._latest = data
                self._sequence += 1
                self._notify()
                await asyncio.sleep(max(0.0, self._period_s - (loop.time() - started)))
        except JetankError as exc:
            logger.error("video: camera failed, video stopped: %s", exc)
            self._failure = f"camera failed: {exc}"[:120]  # close reasons are short
        except Exception as exc:  # e.g. cv2.error: video is optional, driving must go on
            logger.exception("video: unexpected error, video stopped")
            self._failure = f"video error: {type(exc).__name__}"
        finally:
            self._stopped = True
            self._notify()

    def close(self) -> None:
        """Stop the capture thread without waiting for a read in progress. Idempotent."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _capture(self) -> bytes:
        return self._encode(self._camera.read())

    def _notify(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def serve_viewer(self, connection: ServerConnection) -> None:
        """Send frames to one viewer until it disconnects, the camera fails or the stream stops."""
        self._viewers += 1
        self._has_viewers.set()
        closed = asyncio.ensure_future(connection.wait_closed())
        try:
            seen = self._sequence
            while not self._stopped:
                if self._sequence == seen:
                    changed = asyncio.ensure_future(self._changed.wait())
                    await asyncio.wait({changed, closed}, return_when=asyncio.FIRST_COMPLETED)
                    changed.cancel()
                    if closed.done():
                        return
                    continue
                seen, data = self._sequence, self._latest
                assert data is not None  # a sequence change always comes with a frame
                await connection.send(data)
            if self._failure is not None:
                await connection.close(CLOSE_CAMERA_FAILED, self._failure)
        except ConnectionClosed:
            pass
        finally:
            closed.cancel()
            self._viewers -= 1
            if self._viewers == 0:
                self._has_viewers.clear()
