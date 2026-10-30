"""
Port for a small text display (the JETANK's OLED).

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.status import StatusScreen


@runtime_checkable
class Display(Protocol):
    """Shows lines of text.

    Implementations must make ``close()`` idempotent and leave the display blank.
    """

    @property
    def columns(self) -> int:
        """Characters that fit on one line."""
        ...

    @property
    def rows(self) -> int:
        """Lines that fit on the display."""
        ...

    def show(self, screen: StatusScreen) -> None:
        """Replace the display contents with ``screen``."""
        ...

    def clear(self) -> None:
        """Blank the display."""
        ...

    def close(self) -> None:
        """Blank the display, switch it off and release the underlying bus."""
        ...
