"""Protocols that controllers depend on; drivers implement them structurally."""

from jetank.interfaces.camera import Camera
from jetank.interfaces.clock import Clock
from jetank.interfaces.detector import Detector
from jetank.interfaces.display import Display
from jetank.interfaces.events import EventSink
from jetank.interfaces.host import HostInfo
from jetank.interfaces.i2c import I2CBus
from jetank.interfaces.inference import InferenceEngine
from jetank.interfaces.motors import MotorDriver
from jetank.interfaces.power import PowerMonitor

__all__ = [
    "Camera",
    "Clock",
    "Detector",
    "Display",
    "EventSink",
    "HostInfo",
    "I2CBus",
    "InferenceEngine",
    "MotorDriver",
    "PowerMonitor",
]
