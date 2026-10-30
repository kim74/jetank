"""
Hostname and IP address of the computer the software runs on.

Layer: drivers
"""

from __future__ import annotations

import socket
from typing import Final

#: Documentation address (RFC 5737). Connecting a UDP socket only selects a route; no
#: packet is sent, so this works without internet access as long as a default route exists.
_PROBE_ADDRESS: Final = ("192.0.2.1", 9)


class SystemHostInfo:
    """:class:`~jetank.interfaces.HostInfo` for this machine."""

    def hostname(self) -> str:
        return socket.gethostname()

    def ip_address(self) -> str | None:
        """The local address of the default route, or ``None`` without one."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect(_PROBE_ADDRESS)
            except OSError:
                return None
            address: str = probe.getsockname()[0]
        return address
