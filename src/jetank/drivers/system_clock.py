"""
Real monotonic clock.

Layer: drivers
"""

from __future__ import annotations

import time


class SystemClock:
    """:class:`~jetank.interfaces.Clock` backed by :func:`time.monotonic`."""

    def monotonic(self) -> float:
        return time.monotonic()
