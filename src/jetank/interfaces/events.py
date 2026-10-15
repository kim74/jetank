"""
Port for storing intruder events.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.detection import IntruderEvent
    from jetank.domain.models import Frame


@runtime_checkable
class EventSink(Protocol):
    def record(self, event: IntruderEvent, frame: Frame) -> None:
        """Persist or forward ``event`` together with the frame it was seen in."""
        ...

    def close(self) -> None:
        """Flush and release resources. Must be idempotent."""
        ...
