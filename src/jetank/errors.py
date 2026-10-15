"""
Exception hierarchy for the jetank package.

Layer: domain
"""

from __future__ import annotations


class JetankError(Exception):
    """Base class for every error raised by jetank."""


class ConfigurationError(JetankError):
    """Configuration is missing, malformed or inconsistent."""


class HardwareError(JetankError):
    """A hardware device could not be reached or reported a failure."""


class CameraError(HardwareError):
    """The camera could not be opened or failed to deliver a frame."""


class DriverClosedError(JetankError):
    """An operation was attempted on a driver that has already been closed."""
