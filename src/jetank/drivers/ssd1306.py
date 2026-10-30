"""
Driver for the Solomon SSD1306 128-column monochrome OLED controller over I2C.

Layer: drivers

Every I2C write starts with a control byte that says whether the following bytes are
commands or display RAM data. The frame is written in horizontal addressing mode: page 0
(rows 0-7) left to right, then page 1, and so on; bit 0 of each byte is the top row.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

import numpy as np

from jetank.drivers.text_render import render_lines, text_grid
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from types import TracebackType

    from numpy.typing import NDArray

    from jetank.config.settings import DisplayConfig
    from jetank.domain.status import StatusScreen
    from jetank.interfaces.i2c import I2CBus

logger = logging.getLogger(__name__)

WIDTH_PX: Final = 128
CONTROL_COMMAND: Final = 0x00
CONTROL_DATA: Final = 0x40
#: SMBus block transfers carry at most 32 bytes.
MAX_BLOCK: Final = 32

# Commands (datasheet section 9)
DISPLAY_OFF: Final = 0xAE
DISPLAY_ON: Final = 0xAF
SET_COLUMN_RANGE: Final = 0x21
SET_PAGE_RANGE: Final = 0x22


def init_commands(height_px: int) -> list[int]:
    """Power-on sequence for a panel with the internal charge pump (Adafruit-style modules).

    Leaves the display off; it is switched on once RAM has been cleared.

    Raises:
        ValueError: ``height_px`` is not 32 or 64.
    """
    if height_px not in (32, 64):
        raise ValueError(f"panel height must be 32 or 64, got {height_px}")
    com_pins, contrast = (0x02, 0x8F) if height_px == 32 else (0x12, 0xCF)
    return [
        DISPLAY_OFF,
        0xD5, 0x80,              # clock divide ratio / oscillator frequency
        0xA8, height_px - 1,     # multiplex ratio
        0xD3, 0x00,              # display offset
        0x40,                    # start line 0
        0x8D, 0x14,              # charge pump on
        0x20, 0x00,              # horizontal addressing mode
        0xA1,                    # column 127 mapped to SEG0 (mirror for the module's wiring)
        0xC8,                    # scan COM outputs in reverse
        0xDA, com_pins,          # COM pin configuration
        0x81, contrast,          # contrast
        0xD9, 0xF1,              # pre-charge period
        0xDB, 0x40,              # VCOMH deselect level
        0xA4,                    # show RAM contents
        0xA6,                    # normal (not inverted)
    ]  # fmt: skip


def pack_pages(pixels: NDArray[np.bool_]) -> bytes:
    """Convert a ``(height, width)`` pixel array to SSD1306 display RAM bytes.

    Raises:
        ValueError: The array is not 2-D or its height is not a multiple of 8.
    """
    if pixels.ndim != 2 or pixels.shape[0] % 8 != 0:
        raise ValueError(f"pixels must be 2-D with a height divisible by 8, got {pixels.shape}")
    height, width = pixels.shape
    bits = pixels.reshape(height // 8, 8, width).astype(np.uint8)
    weights = (1 << np.arange(8, dtype=np.uint8)).reshape(1, 8, 1)
    return bytes((bits * weights).sum(axis=1, dtype=np.uint8).tobytes())


class Ssd1306Display:
    """:class:`~jetank.interfaces.Display` on an SSD1306 panel, 128 pixels wide.

    Use :meth:`open` to create one on a real bus. ``bus``, if given as ``owned_bus``, is
    closed by :meth:`close`.

    Args:
        bus: I2C bus the panel is attached to.
        address: 7-bit I2C address.
        height_px: Panel height, 32 or 64.
        owned_bus: Pass ``bus`` again if this driver should close it.

    Raises:
        HardwareError: The panel did not respond during initialisation.
    """

    def __init__(
        self, bus: I2CBus, address: int, height_px: int, owned_bus: I2CBus | None = None
    ) -> None:
        self._bus = bus
        self._address = address
        self._height_px = height_px
        self._columns, self._rows = text_grid(WIDTH_PX, height_px)
        self._owned_bus = owned_bus
        self._closed = False
        self._command(init_commands(height_px))
        self._write_frame(self._blank())
        self._command([DISPLAY_ON])
        logger.info("SSD1306 %dx%d at 0x%02X initialised", WIDTH_PX, height_px, address)

    @classmethod
    def open(cls, config: DisplayConfig) -> Ssd1306Display:
        """Open the I2C bus from ``config`` and initialise the panel.

        Raises:
            HardwareError: The bus cannot be opened or the panel does not respond.
        """
        from smbus2 import SMBus

        try:
            bus = SMBus(config.i2c_bus)
        except OSError as exc:
            raise HardwareError(f"cannot open /dev/i2c-{config.i2c_bus}: {exc}") from exc
        try:
            return cls(bus, config.i2c_address, config.height, owned_bus=bus)
        except BaseException:
            bus.close()
            raise

    def __enter__(self) -> Ssd1306Display:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    def show(self, screen: StatusScreen) -> None:
        self._ensure_open()
        self._write_frame(pack_pages(render_lines(screen.lines, WIDTH_PX, self._height_px)))

    def clear(self) -> None:
        self._ensure_open()
        self._write_frame(self._blank())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._write_frame(self._blank())
            self._command([DISPLAY_OFF])
        finally:
            if self._owned_bus is not None:
                self._owned_bus.close()

    def _blank(self) -> bytes:
        return bytes(WIDTH_PX * self._height_px // 8)

    def _write_frame(self, frame: bytes) -> None:
        self._command(
            [SET_COLUMN_RANGE, 0, WIDTH_PX - 1, SET_PAGE_RANGE, 0, self._height_px // 8 - 1]
        )
        self._transfer(CONTROL_DATA, frame)

    def _command(self, commands: list[int]) -> None:
        self._transfer(CONTROL_COMMAND, bytes(commands))

    def _transfer(self, control: int, payload: bytes) -> None:
        try:
            for start in range(0, len(payload), MAX_BLOCK):
                chunk = list(payload[start : start + MAX_BLOCK])
                self._bus.write_i2c_block_data(self._address, control, chunk)
        except OSError as exc:
            raise HardwareError(f"SSD1306 at 0x{self._address:02X} not responding: {exc}") from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("display is closed")
