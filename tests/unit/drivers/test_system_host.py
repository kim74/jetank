from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING

from jetank.drivers.system_host import SystemHostInfo

if TYPE_CHECKING:
    import pytest


def test_hostname_is_the_system_hostname() -> None:
    assert SystemHostInfo().hostname() == socket.gethostname()


def test_ip_address_is_an_ipv4_address_or_none() -> None:
    address = SystemHostInfo().ip_address()
    if address is not None:
        assert ipaddress.ip_address(address).version == 4


def test_no_route_gives_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable(_self: socket.socket, _address: object) -> None:
        raise OSError(101, "Network is unreachable")

    monkeypatch.setattr(socket.socket, "connect", unreachable)
    assert SystemHostInfo().ip_address() is None
