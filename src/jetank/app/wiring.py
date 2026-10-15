"""
Composition root: the only place that decides which concrete driver backs each interface.

Layer: app
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jetank.control.drive import DriveController
from jetank.drivers.fake.camera import FakeCamera
from jetank.drivers.fake.motors import FakeMotorDriver
from jetank.drivers.system_clock import SystemClock

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.settings import CameraConfig, JetankSettings, MotorsConfig
    from jetank.interfaces.camera import Camera
    from jetank.interfaces.clock import Clock
    from jetank.interfaces.motors import MotorDriver


def build_motor_driver(config: MotorsConfig) -> MotorDriver:
    """Create the track motor driver selected by ``config.backend``."""
    if config.backend == "fake":
        return FakeMotorDriver()
    from jetank.drivers.tb6612_tracks import Tb6612TrackDriver

    return Tb6612TrackDriver.open(config)


def build_camera(config: CameraConfig, clock: Clock) -> Camera:
    """Create the camera selected by ``config.backend``."""
    if config.backend == "fake":
        return FakeCamera(config.width, config.height, clock)
    from jetank.drivers.gst_camera import GstCamera

    return GstCamera(config, clock)


@dataclass
class Robot:
    """All subsystems of one robot, closed together (motors stopped first)."""

    motors: MotorDriver
    drive: DriveController
    camera: Camera
    _stack: contextlib.ExitStack = field(repr=False)

    def __enter__(self) -> Robot:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._stack.close()


def build_robot(settings: JetankSettings, clock: Clock | None = None) -> Robot:
    """Wire every subsystem from settings. If any part fails, those already built are closed."""
    clock = clock if clock is not None else SystemClock()
    with contextlib.ExitStack() as stack:
        # Callbacks run in reverse order: the camera closes first, motors last.
        motors = build_motor_driver(settings.motors)
        stack.callback(motors.close)
        drive = DriveController(motors, clock, settings.drive)
        stack.callback(drive.stop)
        camera = build_camera(settings.camera, clock)
        stack.callback(camera.close)
        return Robot(motors=motors, drive=drive, camera=camera, _stack=stack.pop_all())
