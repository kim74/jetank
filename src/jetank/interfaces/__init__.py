"""Protocols that controllers depend on; drivers implement them structurally."""

from jetank.interfaces.camera import Camera
from jetank.interfaces.clock import Clock
from jetank.interfaces.detector import Detector
from jetank.interfaces.events import EventSink
from jetank.interfaces.i2c import I2CBus
from jetank.interfaces.inference import InferenceEngine
from jetank.interfaces.motors import MotorDriver

__all__ = [
    "Camera",
    "Clock",
    "Detector",
    "EventSink",
    "I2CBus",
    "InferenceEngine",
    "MotorDriver",
]
