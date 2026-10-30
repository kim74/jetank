"""
Host information with settable values.

Layer: drivers (fake)
"""

from __future__ import annotations


class FakeHostInfo:
    """:class:`~jetank.interfaces.HostInfo` returning fixed, mutable values."""

    def __init__(self, hostname: str = "jetank", ip_address: str | None = "192.0.2.10") -> None:
        self.name = hostname
        self.address = ip_address

    def hostname(self) -> str:
        return self.name

    def ip_address(self) -> str | None:
        return self.address
