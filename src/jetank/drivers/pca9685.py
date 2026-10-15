"""
Register-level driver for the NXP PCA9685 16-channel, 12-bit PWM controller.

Layer: drivers
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Final

from jetank.errors import HardwareError

if TYPE_CHECKING:
    from collections.abc import Callable

    from jetank.interfaces.i2c import I2CBus

logger = logging.getLogger(__name__)

# Registers (datasheet section 7.3)
MODE1: Final = 0x00
MODE2: Final = 0x01
LED0_ON_L: Final = 0x06
ALL_LED_ON_L: Final = 0xFA
PRESCALE: Final = 0xFE

# MODE1 / MODE2 bits
MODE1_RESTART: Final = 0x80
MODE1_AUTO_INCREMENT: Final = 0x20
MODE1_SLEEP: Final = 0x10
MODE2_OUTDRV: Final = 0x04

# The 4th byte of an LEDn block carries the "full off" flag (bit 4 of OFF_H);
# the 2nd byte carries the "full on" flag (bit 4 of ON_H).
FULL_FLAG: Final = 0x10

CHANNELS: Final = 16
OSCILLATOR_HZ: Final = 25_000_000
RESOLUTION: Final = 4096
OSCILLATOR_SETTLE_S: Final = 0.005


def prescale_for(frequency_hz: int) -> int:
    """PRESCALE register value for the requested PWM frequency (datasheet eq. 1)."""
    if frequency_hz <= 0:
        raise ValueError("frequency must be positive")
    value = round(OSCILLATOR_HZ / (RESOLUTION * frequency_hz)) - 1
    return max(3, min(0xFF, value))


def duty_to_block(duty: float) -> list[int]:
    """Encode a duty cycle in [0, 1] as the 4 bytes of an LEDn_ON_L..LEDn_OFF_H block.

    Exactly 0 and 1 use the chip's full-off / full-on flags so the output is a clean
    constant level with no residual pulse.
    """
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"duty cycle must be within [0, 1], got {duty}")
    if duty == 0.0:
        return [0x00, 0x00, 0x00, FULL_FLAG]
    if duty == 1.0:
        return [0x00, FULL_FLAG, 0x00, 0x00]
    off = min(RESOLUTION - 1, max(1, round(duty * RESOLUTION)))
    return [0x00, 0x00, off & 0xFF, off >> 8]


class Pca9685:
    """PCA9685 on a shared I2C bus. Does not own (or close) the bus.

    Args:
        bus: I2C bus the chip is attached to.
        address: 7-bit I2C address.
        frequency_hz: PWM output frequency.
        sleep: Delay function, injectable so tests do not wait.

    Raises:
        HardwareError: The chip did not respond during initialisation.
    """

    def __init__(
        self,
        bus: I2CBus,
        address: int,
        frequency_hz: int,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._bus = bus
        self._address = address
        self._sleep = sleep
        try:
            self._initialise(frequency_hz)
        except OSError as exc:
            raise HardwareError(
                f"PCA9685 not responding at address 0x{address:02X}: {exc}"
            ) from exc
        logger.info("PCA9685 at 0x%02X initialised at %d Hz", address, frequency_hz)

    def _initialise(self, frequency_hz: int) -> None:
        self._bus.write_i2c_block_data(self._address, ALL_LED_ON_L, [0, 0, 0, FULL_FLAG])
        self._bus.write_byte_data(self._address, MODE2, MODE2_OUTDRV)
        self._bus.write_byte_data(self._address, MODE1, MODE1_AUTO_INCREMENT)
        self._sleep(OSCILLATOR_SETTLE_S)
        # Prescale can only be written while the oscillator is asleep.
        awake = self._bus.read_byte_data(self._address, MODE1) & ~(MODE1_RESTART | MODE1_SLEEP)
        self._bus.write_byte_data(self._address, MODE1, awake | MODE1_SLEEP)
        self._bus.write_byte_data(self._address, PRESCALE, prescale_for(frequency_hz))
        self._bus.write_byte_data(self._address, MODE1, awake)
        self._sleep(OSCILLATOR_SETTLE_S)
        self._bus.write_byte_data(self._address, MODE1, awake | MODE1_RESTART)

    def set_duty(self, channel: int, duty: float) -> None:
        """Set one channel's duty cycle in [0, 1].

        Raises:
            ValueError: Channel or duty cycle out of range.
            HardwareError: The I2C write failed.
        """
        if not 0 <= channel < CHANNELS:
            raise ValueError(f"channel must be within [0, {CHANNELS - 1}], got {channel}")
        block = duty_to_block(duty)
        try:
            self._bus.write_i2c_block_data(self._address, LED0_ON_L + 4 * channel, block)
        except OSError as exc:
            raise HardwareError(f"PCA9685 write to channel {channel} failed: {exc}") from exc

    def all_off(self) -> None:
        """Drive every channel fully low.

        Raises:
            HardwareError: The I2C write failed.
        """
        try:
            self._bus.write_i2c_block_data(self._address, ALL_LED_ON_L, [0, 0, 0, FULL_FLAG])
        except OSError as exc:
            raise HardwareError(f"PCA9685 all-off failed: {exc}") from exc
