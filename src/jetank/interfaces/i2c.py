"""
Port for raw I2C register access. ``smbus2.SMBus`` satisfies it structurally.

Layer: interfaces
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class I2CBus(Protocol):
    def read_byte_data(self, i2c_addr: int, register: int) -> int: ...

    def write_byte_data(self, i2c_addr: int, register: int, value: int) -> None: ...

    def write_i2c_block_data(self, i2c_addr: int, register: int, data: list[int]) -> None: ...

    def close(self) -> None: ...
