"""
Port for facts about the computer the software runs on.

Layer: interfaces
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HostInfo(Protocol):
    def hostname(self) -> str:
        """Network name of this computer."""
        ...

    def ip_address(self) -> str | None:
        """Address used to reach the network, or ``None`` when not connected."""
        ...
