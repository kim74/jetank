from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jetank.domain.status import StatusScreen
from jetank.drivers.ssd1306 import (
    CONTROL_COMMAND,
    CONTROL_DATA,
    DISPLAY_OFF,
    DISPLAY_ON,
    Ssd1306Display,
    init_commands,
    pack_pages,
)
from jetank.drivers.text_render import render_lines, text_grid
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from jetank.drivers.fake import FakeI2CBus

ADDRESS = 0x3C


def commands(bus: FakeI2CBus) -> list[int]:
    return [b for _a, control, data in bus.block_writes if control == CONTROL_COMMAND for b in data]


def last_frame(bus: FakeI2CBus, frame_bytes: int) -> bytes:
    """The most recent full frame written to display RAM."""
    data = [b for _a, control, chunk in bus.block_writes if control == CONTROL_DATA for b in chunk]
    return bytes(data[-frame_bytes:])


def unpack_pages(frame: bytes, height: int, width: int = 128) -> np.ndarray:
    pages = np.frombuffer(frame, dtype=np.uint8).reshape(height // 8, 1, width)
    bits = (pages >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)) & 1
    return bits.reshape(height, width).astype(bool)


def test_pack_sets_bit0_for_top_row_of_each_page() -> None:
    pixels = np.zeros((16, 128), dtype=bool)
    pixels[0, 0] = True  # page 0, column 0, bit 0
    pixels[7, 1] = True  # page 0, column 1, bit 7
    pixels[8, 127] = True  # page 1, column 127, bit 0
    frame = pack_pages(pixels)
    assert len(frame) == 256
    assert frame[0] == 0x01
    assert frame[1] == 0x80
    assert frame[128 + 127] == 0x01


@given(arrays(np.bool_, st.sampled_from([(32, 128), (64, 128)])))
def test_pack_roundtrip(pixels: np.ndarray) -> None:
    height = pixels.shape[0]
    assert np.array_equal(unpack_pages(pack_pages(pixels), height), pixels)


@pytest.mark.parametrize("shape", [(30, 128), (128,)])
def test_pack_rejects_bad_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        pack_pages(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize(("height", "multiplex", "com_pins"), [(32, 0x1F, 0x02), (64, 0x3F, 0x12)])
def test_init_sequence_matches_panel_height(height: int, multiplex: int, com_pins: int) -> None:
    sequence = init_commands(height)
    assert sequence[0] == DISPLAY_OFF
    assert sequence[sequence.index(0xA8) + 1] == multiplex
    assert sequence[sequence.index(0xDA) + 1] == com_pins
    assert sequence[sequence.index(0x8D) + 1] == 0x14  # charge pump on


def test_init_rejects_unsupported_height() -> None:
    with pytest.raises(ValueError, match="32 or 64"):
        init_commands(48)


def test_open_clears_ram_before_switching_on(i2c_bus: FakeI2CBus) -> None:
    Ssd1306Display(i2c_bus, ADDRESS, 32)
    assert commands(i2c_bus)[0] == DISPLAY_OFF
    assert commands(i2c_bus)[-1] == DISPLAY_ON
    assert last_frame(i2c_bus, 512) == bytes(512)
    assert all(address == ADDRESS for address, _c, _d in i2c_bus.block_writes)


def test_transfers_respect_smbus_block_limit(i2c_bus: FakeI2CBus) -> None:
    # FakeI2CBus raises on blocks over 32 bytes, so a full frame must be chunked.
    display = Ssd1306Display(i2c_bus, ADDRESS, 64)
    display.show(StatusScreen(lines=("hello",)))
    assert max(len(data) for _a, _c, data in i2c_bus.block_writes) <= 32


def test_show_writes_rendered_text(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32)
    lines = ("jetank", "192.168.1.42", "bat 11.9V", "idle")
    display.show(StatusScreen(lines=lines))
    shown = unpack_pages(last_frame(i2c_bus, 512), 32)
    assert np.array_equal(shown, render_lines(lines, 128, 32))
    assert shown.any()


def test_grid_for_128x32_is_21_by_4(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32)
    assert (display.columns, display.rows) == (21, 4) == text_grid(128, 32)


def test_clear_blanks_ram(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32)
    display.show(StatusScreen(lines=("x",)))
    display.clear()
    assert last_frame(i2c_bus, 512) == bytes(512)


def test_close_blanks_switches_off_and_closes_owned_bus(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32, owned_bus=i2c_bus)
    display.show(StatusScreen(lines=("x",)))
    with display:
        pass
    assert last_frame(i2c_bus, 512) == bytes(512)
    assert commands(i2c_bus)[-1] == DISPLAY_OFF
    assert i2c_bus.closed
    display.close()  # idempotent


def test_close_leaves_shared_bus_open(i2c_bus: FakeI2CBus) -> None:
    Ssd1306Display(i2c_bus, ADDRESS, 32).close()
    assert not i2c_bus.closed


def test_use_after_close_raises(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32)
    display.close()
    with pytest.raises(DriverClosedError):
        display.show(StatusScreen(lines=("x",)))
    with pytest.raises(DriverClosedError):
        display.clear()


def test_missing_panel_raises_hardware_error(i2c_bus: FakeI2CBus) -> None:
    i2c_bus.fail_with = OSError(121, "Remote I/O error")
    with pytest.raises(HardwareError, match="0x3C not responding"):
        Ssd1306Display(i2c_bus, ADDRESS, 32)


def test_close_releases_owned_bus_even_if_panel_fails(i2c_bus: FakeI2CBus) -> None:
    display = Ssd1306Display(i2c_bus, ADDRESS, 32, owned_bus=i2c_bus)
    i2c_bus.fail_with = OSError(121, "Remote I/O error")
    with pytest.raises(HardwareError):
        display.close()
    assert i2c_bus.closed


def test_text_beyond_the_panel_is_clipped() -> None:
    pixels = render_lines(["W" * 40] * 8, 128, 32)
    assert pixels.shape == (32, 128)
