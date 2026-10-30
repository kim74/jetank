"""
In-memory display that records every screen shown.

Layer: drivers (fake)
"""

from __future__ import annotations

from jetank.domain.status import StatusScreen
from jetank.errors import DriverClosedError, HardwareError


class FakeDisplay:
    """:class:`~jetank.interfaces.Display` that stores screens instead of drawing them.

    Attributes:
        history: Every screen shown, in order (``clear()`` records an empty screen).
        fail_with: If set, ``show`` and ``clear`` raise this (to simulate a broken display).
    """

    def __init__(self, columns: int = 21, rows: int = 4) -> None:
        self._columns = columns
        self._rows = rows
        self.history: list[StatusScreen] = []
        self.fail_with: HardwareError | None = None
        self.closed = False

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def current(self) -> StatusScreen:
        """The screen on display now (empty if nothing was shown yet)."""
        return self.history[-1] if self.history else StatusScreen(lines=())

    def show(self, screen: StatusScreen) -> None:
        self._check()
        self.history.append(screen)

    def clear(self) -> None:
        self._check()
        self.history.append(StatusScreen(lines=()))

    def close(self) -> None:
        if not self.closed:
            self.history.append(StatusScreen(lines=()))
            self.closed = True

    def _check(self) -> None:
        if self.closed:
            raise DriverClosedError("display is closed")
        if self.fail_with is not None:
            raise self.fail_with
