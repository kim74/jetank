"""Fixtures shared by the hardware tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jetank.app.signals import raise_on_termination_signals

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _stop_on_termination_signal() -> Iterator[None]:
    """A dropped SSH session (SIGHUP) or ``kill`` (SIGTERM) aborts the run like Ctrl-C.

    Without this, Python exits on those signals without running any teardown, and the motor
    driver would never stop the tracks.
    """
    with raise_on_termination_signals():
        yield
