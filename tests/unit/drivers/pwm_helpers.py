"""Helpers to decode PCA9685 register contents from a FakeI2CBus."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jetank.drivers.pca9685 import FULL_FLAG, LED0_ON_L, RESOLUTION

if TYPE_CHECKING:
    from jetank.drivers.fake import FakeI2CBus


def channel_duty(bus: FakeI2CBus, address: int, channel: int) -> float:
    """Decode the duty cycle a PCA9685 channel is currently configured for."""
    base = LED0_ON_L + 4 * channel
    regs = bus.registers[address]
    on_l, on_h, off_l, off_h = (regs.get(base + i, 0) for i in range(4))
    if off_h & FULL_FLAG:
        return 0.0
    if on_h & FULL_FLAG:
        return 1.0
    assert (on_l, on_h & 0x0F) == (0, 0), "driver only uses ON=0"
    return ((off_h & 0x0F) << 8 | off_l) / RESOLUTION
