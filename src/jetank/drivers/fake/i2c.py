"""
Register-level I2C bus simulator for testing chip drivers without hardware.

Layer: drivers (fake)
"""

from __future__ import annotations

from collections import defaultdict

from jetank.errors import DriverClosedError


class FakeI2CBus:
    """:class:`~jetank.interfaces.I2CBus` storing registers per device address.

    Block writes fill consecutive registers, matching chips with register auto-increment.

    Attributes:
        registers: ``registers[address][register]`` holds the last value written.
        writes: Log of ``(address, register, value)`` for every byte written, in order.
        fail_with: If set, every access raises this exception (to simulate a missing device).
    """

    def __init__(self) -> None:
        self.registers: defaultdict[int, dict[int, int]] = defaultdict(dict)
        self.writes: list[tuple[int, int, int]] = []
        self.fail_with: OSError | None = None
        self.closed = False

    def _check(self) -> None:
        if self.closed:
            raise DriverClosedError("I2C bus is closed")
        if self.fail_with is not None:
            raise self.fail_with

    def read_byte_data(self, i2c_addr: int, register: int) -> int:
        self._check()
        return self.registers[i2c_addr].get(register, 0)

    def write_byte_data(self, i2c_addr: int, register: int, value: int) -> None:
        self._check()
        if not 0 <= value <= 0xFF:
            raise ValueError(f"byte value out of range: {value}")
        self.registers[i2c_addr][register] = value
        self.writes.append((i2c_addr, register, value))

    def write_i2c_block_data(self, i2c_addr: int, register: int, data: list[int]) -> None:
        for offset, value in enumerate(data):
            self.write_byte_data(i2c_addr, register + offset, value)

    def close(self) -> None:
        self.closed = True
