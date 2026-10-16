"""
Turn termination signals into an exception so the robot is left in a safe state.

Layer: app

Python's default action for SIGHUP and SIGTERM ends the process without running ``finally``
blocks or ``__exit__`` methods, so motors would keep their last command. Inside
:func:`raise_on_termination_signals` those signals raise :class:`TerminationRequested`
instead, which unwinds the stack exactly like Ctrl-C does.
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from jetank.errors import TerminationRequested

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

logger = logging.getLogger(__name__)

TERMINATION_SIGNALS: Final = (signal.SIGHUP, signal.SIGTERM)


def _raise_termination(signum: int, _frame: FrameType | None) -> None:
    name = signal.Signals(signum).name
    logger.warning("%s received, unwinding to a safe state", name)
    raise TerminationRequested(name)


@contextmanager
def raise_on_termination_signals() -> Iterator[None]:
    """Raise :class:`TerminationRequested` on SIGHUP/SIGTERM while the block runs.

    The previous handlers are restored on exit. Must be entered from the main thread, as
    Python only allows signal handlers to be installed there.

    Raises:
        ValueError: Called from a thread other than the main thread.
    """
    previous = {signum: signal.signal(signum, _raise_termination) for signum in TERMINATION_SIGNALS}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
