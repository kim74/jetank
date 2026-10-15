from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jetank.drivers.fake import FakeI2CBus
from jetank.drivers.pca9685 import (
    MODE1,
    MODE1_AUTO_INCREMENT,
    MODE1_RESTART,
    MODE1_SLEEP,
    MODE2,
    MODE2_OUTDRV,
    PRESCALE,
    Pca9685,
    duty_to_block,
    prescale_for,
)
from jetank.errors import HardwareError
from tests.unit.drivers.pwm_helpers import channel_duty

ADDRESS = 0x60


def make_chip(bus: FakeI2CBus, frequency_hz: int = 1600) -> Pca9685:
    return Pca9685(bus, ADDRESS, frequency_hz, sleep=lambda _s: None)


@pytest.mark.parametrize(
    ("frequency_hz", "expected"),
    [(1600, 3), (1000, 5), (50, 121), (24, 253), (1526, 3), (100_000, 3)],
)
def test_prescale_matches_datasheet_formula(frequency_hz: int, expected: int) -> None:
    assert prescale_for(frequency_hz) == expected


def test_prescale_rejects_non_positive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        prescale_for(0)


def test_duty_extremes_use_full_flags() -> None:
    assert duty_to_block(0.0) == [0, 0, 0, 0x10]
    assert duty_to_block(1.0) == [0, 0x10, 0, 0]


def test_duty_half_is_2048_ticks() -> None:
    assert duty_to_block(0.5) == [0, 0, 0x00, 0x08]


@pytest.mark.parametrize("duty", [-0.01, 1.01])
def test_duty_out_of_range_rejected(duty: float) -> None:
    with pytest.raises(ValueError, match="duty"):
        duty_to_block(duty)


@given(st.floats(min_value=0.0, max_value=1.0))
def test_duty_roundtrip_within_one_tick(duty: float) -> None:
    bus = FakeI2CBus()
    bus.write_i2c_block_data(ADDRESS, 0x06, duty_to_block(duty))
    assert channel_duty(bus, ADDRESS, 0) == pytest.approx(duty, abs=1 / 4096)


def test_init_configures_frequency_and_wakes_chip(i2c_bus: FakeI2CBus) -> None:
    make_chip(i2c_bus, frequency_hz=50)
    regs = i2c_bus.registers[ADDRESS]
    assert regs[PRESCALE] == 121
    assert regs[MODE2] == MODE2_OUTDRV
    assert regs[MODE1] & MODE1_RESTART
    assert regs[MODE1] & MODE1_AUTO_INCREMENT
    assert not regs[MODE1] & MODE1_SLEEP


def test_prescale_written_while_asleep(i2c_bus: FakeI2CBus) -> None:
    make_chip(i2c_bus)
    mode1_before_prescale = None
    for _addr, register, value in i2c_bus.writes:
        if register == MODE1:
            mode1_before_prescale = value
        if register == PRESCALE:
            break
    assert mode1_before_prescale is not None
    assert mode1_before_prescale & MODE1_SLEEP


def test_init_turns_all_outputs_off(i2c_bus: FakeI2CBus) -> None:
    make_chip(i2c_bus)
    first_write = i2c_bus.writes[0]
    assert first_write[1] == 0xFA  # ALL_LED_ON_L block is the very first thing written


def test_set_duty_targets_the_right_channel(i2c_bus: FakeI2CBus) -> None:
    chip = make_chip(i2c_bus)
    chip.set_duty(5, 0.25)
    assert channel_duty(i2c_bus, ADDRESS, 5) == pytest.approx(0.25)
    assert channel_duty(i2c_bus, ADDRESS, 4) == 0.0


@pytest.mark.parametrize("channel", [-1, 16])
def test_set_duty_rejects_invalid_channel(i2c_bus: FakeI2CBus, channel: int) -> None:
    with pytest.raises(ValueError, match="channel"):
        make_chip(i2c_bus).set_duty(channel, 0.5)


def test_missing_chip_raises_hardware_error(i2c_bus: FakeI2CBus) -> None:
    i2c_bus.fail_with = OSError(121, "Remote I/O error")
    with pytest.raises(HardwareError, match="0x60"):
        make_chip(i2c_bus)


def test_bus_failure_after_init_raises_hardware_error(i2c_bus: FakeI2CBus) -> None:
    chip = make_chip(i2c_bus)
    i2c_bus.fail_with = OSError(5, "I/O error")
    with pytest.raises(HardwareError, match="channel 3"):
        chip.set_duty(3, 0.5)
    with pytest.raises(HardwareError, match="all-off"):
        chip.all_off()
