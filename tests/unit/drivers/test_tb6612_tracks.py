from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from jetank.config.settings import MotorChannelConfig, MotorsConfig
from jetank.domain.models import TrackSpeeds
from jetank.drivers.fake import FakeI2CBus
from jetank.drivers.pca9685 import LED0_ON_L, Pca9685
from jetank.drivers.tb6612_tracks import BridgeOutputs, Tb6612TrackDriver, bridge_outputs
from jetank.errors import DriverClosedError, HardwareError
from jetank.interfaces import MotorDriver
from tests.unit.drivers.pwm_helpers import channel_duty

if TYPE_CHECKING:
    from collections.abc import Iterator

ADDRESS = 0x60
SPEEDS = st.floats(min_value=-1.0, max_value=1.0)


def make_driver(bus: FakeI2CBus, config: MotorsConfig, *, owns_bus: bool) -> Tb6612TrackDriver:
    pwm = Pca9685(bus, ADDRESS, config.pwm_frequency_hz, sleep=lambda _s: None)
    return Tb6612TrackDriver(pwm, config, bus=bus if owns_bus else None)


def outputs(bus: FakeI2CBus, channel: MotorChannelConfig) -> tuple[float, float, float]:
    """Current ``(pwm, forward, reverse)`` duty cycles of one motor's channels."""
    return (
        channel_duty(bus, ADDRESS, channel.pwm_channel),
        channel_duty(bus, ADDRESS, channel.forward_channel),
        channel_duty(bus, ADDRESS, channel.reverse_channel),
    )


def replay_states(bus: FakeI2CBus) -> Iterator[FakeI2CBus]:
    """Yield the chip state after every byte in ``bus.writes``, as seen by the outputs."""
    replay = FakeI2CBus()
    for address, register, value in bus.writes:
        replay.write_byte_data(address, register, value)
        yield replay


class TestBridgeOutputs:
    plain = MotorChannelConfig(pwm_channel=8, forward_channel=9, reverse_channel=10)

    def test_forward_raises_forward_input(self) -> None:
        assert bridge_outputs(0.4, self.plain) == BridgeOutputs(
            pwm_duty=0.4, forward=True, reverse=False
        )

    def test_reverse_raises_reverse_input(self) -> None:
        assert bridge_outputs(-0.4, self.plain) == BridgeOutputs(
            pwm_duty=0.4, forward=False, reverse=True
        )

    def test_zero_coasts(self) -> None:
        assert bridge_outputs(0.0, self.plain) == BridgeOutputs(
            pwm_duty=0.0, forward=False, reverse=False
        )

    def test_inverted_swaps_direction(self) -> None:
        inverted = self.plain.model_copy(update={"inverted": True})
        assert bridge_outputs(0.4, inverted).reverse

    def test_trim_scales_and_result_is_clamped(self) -> None:
        trimmed = self.plain.model_copy(update={"trim": 1.2})
        assert bridge_outputs(0.5, trimmed).pwm_duty == pytest.approx(0.6)
        assert bridge_outputs(-1.0, trimmed).pwm_duty == 1.0

    def test_both_directions_high_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both be high"):
            BridgeOutputs(pwm_duty=0.5, forward=True, reverse=True)

    def test_duty_without_direction_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="needs a direction"):
            BridgeOutputs(pwm_duty=0.5, forward=False, reverse=False)

    @given(SPEEDS, st.floats(min_value=0.01, max_value=1.5), st.booleans())
    def test_duty_is_bounded_and_direction_follows_sign(
        self, speed: float, trim: float, inverted: bool
    ) -> None:
        channel = self.plain.model_copy(update={"trim": trim, "inverted": inverted})
        result = bridge_outputs(speed, channel)
        effective = speed * trim * (-1.0 if inverted else 1.0)
        assert 0.0 <= result.pwm_duty <= 1.0
        assert result.forward == (effective > 0.0)
        assert result.reverse == (effective < 0.0)


class TestTb6612TrackDriver:
    def test_satisfies_motor_driver_interface(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        assert isinstance(make_driver(i2c_bus, motors_config, owns_bus=False), MotorDriver)

    def test_set_speeds_drives_each_track(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.set_speeds(TrackSpeeds(left=0.5, right=-0.25))
        assert outputs(i2c_bus, motors_config.left) == pytest.approx((0.5, 1.0, 0.0))
        assert outputs(i2c_bus, motors_config.right) == pytest.approx((0.25, 0.0, 1.0))

    def test_default_channels_match_verified_jetank_wiring(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        # Verified 2026-10-16: left PWM 8, forward 9; right PWM 13, forward 12.
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.set_speeds(TrackSpeeds(left=0.5, right=0.5))
        assert [channel_duty(i2c_bus, ADDRESS, ch) for ch in (8, 9, 10)] == [0.5, 1.0, 0.0]
        assert [channel_duty(i2c_bus, ADDRESS, ch) for ch in (13, 12, 11)] == [0.5, 1.0, 0.0]

    def test_direction_change_lowers_inactive_input_first(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        left = motors_config.left
        driver.set_speeds(TrackSpeeds(left=-0.5, right=0.0))
        i2c_bus.writes.clear()
        driver.set_speeds(TrackSpeeds(left=0.5, right=0.0))
        first_register = i2c_bus.writes[0][1]
        assert first_register == LED0_ON_L + 4 * left.reverse_channel

    @given(st.lists(st.tuples(SPEEDS, SPEEDS), min_size=1, max_size=8))
    def test_direction_inputs_are_never_high_together(
        self, commands: list[tuple[float, float]]
    ) -> None:
        bus = FakeI2CBus()
        config = MotorsConfig()
        driver = make_driver(bus, config, owns_bus=False)
        for left, right in commands:
            driver.set_speeds(TrackSpeeds(left=left, right=right))
        driver.stop()
        for state in replay_states(bus):
            for channel in (config.left, config.right):
                _, forward, reverse = outputs(state, channel)
                assert not (forward == 1.0 and reverse == 1.0)

    def test_stop_drives_all_inputs_low(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.set_speeds(TrackSpeeds(left=1.0, right=1.0))
        driver.stop()
        assert outputs(i2c_bus, motors_config.left) == (0.0, 0.0, 0.0)
        assert outputs(i2c_bus, motors_config.right) == (0.0, 0.0, 0.0)

    def test_close_stops_and_closes_owned_bus(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        with make_driver(i2c_bus, motors_config, owns_bus=True) as driver:
            driver.set_speeds(TrackSpeeds(left=0.3, right=0.3))
        assert i2c_bus.closed
        assert outputs(i2c_bus, motors_config.left) == (0.0, 0.0, 0.0)
        assert outputs(i2c_bus, motors_config.right) == (0.0, 0.0, 0.0)

    def test_close_leaves_shared_bus_open_and_is_idempotent(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.close()
        driver.close()
        assert not i2c_bus.closed

    def test_use_after_close_raises(self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.close()
        with pytest.raises(DriverClosedError):
            driver.set_speeds(TrackSpeeds.stopped())
        with pytest.raises(DriverClosedError):
            driver.stop()

    def test_close_still_closes_bus_when_stop_fails(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=True)
        i2c_bus.fail_with = OSError(5, "I/O error")
        with pytest.raises(HardwareError):
            driver.close()
        assert i2c_bus.closed

    def test_open_on_missing_bus_raises_hardware_error(self) -> None:
        pytest.importorskip("smbus2")
        with pytest.raises(HardwareError, match="i2c-250"):
            Tb6612TrackDriver.open(MotorsConfig(i2c_bus=250))
