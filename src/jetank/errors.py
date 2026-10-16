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


class TerminationRequested(KeyboardInterrupt):
    """The process received SIGHUP or SIGTERM (e.g. an SSH session dropped).

    Derives from ``KeyboardInterrupt`` rather than :class:`JetankError` on purpose: like Ctrl-C,
    it must not be swallowed by ``except Exception`` handlers, and it unwinds every ``with``
    block on its way out so drivers leave the robot in a safe state.
    """

    def __init__(self, signal_name: str) -> None:
        super().__init__(f"received {signal_name}")
        self.signal_name = signal_name
