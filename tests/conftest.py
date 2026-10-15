"""Shared fixtures. Everything here is hardware-free."""

from __future__ import annotations

import os

import pytest

from jetank.config.settings import CameraConfig, DriveConfig, MotorsConfig
from jetank.drivers.fake import FakeClock, FakeI2CBus, FakeMotorDriver


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep JETANK_* variables from the developer's shell out of the tests."""
    for name in list(os.environ):
        if name.startswith("JETANK_"):
            monkeypatch.delenv(name)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(start_s=100.0)


@pytest.fixture
def motors() -> FakeMotorDriver:
    return FakeMotorDriver()


@pytest.fixture
def i2c_bus() -> FakeI2CBus:
    return FakeI2CBus()


@pytest.fixture
def motors_config() -> MotorsConfig:
    return MotorsConfig()


@pytest.fixture
def drive_config() -> DriveConfig:
    return DriveConfig(max_speed=0.5, watchdog_timeout_s=0.5)


@pytest.fixture
def camera_config() -> CameraConfig:
    return CameraConfig(width=64, height=48)
