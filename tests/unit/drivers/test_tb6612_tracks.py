from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jetank.config.settings import MotorChannelConfig, MotorsConfig
from jetank.domain.models import TrackSpeeds
from jetank.drivers.pca9685 import LED0_ON_L, Pca9685
from jetank.drivers.tb6612_tracks import Tb6612TrackDriver, channel_duties
from jetank.errors import DriverClosedError, HardwareError
from jetank.interfaces import MotorDriver
from tests.unit.drivers.pwm_helpers import channel_duty

if TYPE_CHECKING:
    from jetank.drivers.fake import FakeI2CBus

ADDRESS = 0x60


def make_driver(bus: FakeI2CBus, config: MotorsConfig, *, owns_bus: bool) -> Tb6612TrackDriver:
    pwm = Pca9685(bus, ADDRESS, config.pwm_frequency_hz, sleep=lambda _s: None)
    return Tb6612TrackDriver(pwm, config, bus=bus if owns_bus else None)


def duties(bus: FakeI2CBus, channel: MotorChannelConfig) -> tuple[float, float]:
    return (
        channel_duty(bus, ADDRESS, channel.pwm_in_a),
        channel_duty(bus, ADDRESS, channel.pwm_in_b),
    )


class TestChannelDuties:
    plain = MotorChannelConfig(pwm_in_a=0, pwm_in_b=1)

    def test_forward_drives_input_a(self) -> None:
        assert channel_duties(0.4, self.plain) == (0.4, 0.0)

    def test_reverse_drives_input_b(self) -> None:
        assert channel_duties(-0.4, self.plain) == (0.0, 0.4)

    def test_zero_coasts(self) -> None:
        assert channel_duties(0.0, self.plain) == (0.0, 0.0)

    def test_inverted_swaps_direction(self) -> None:
        inverted = MotorChannelConfig(pwm_in_a=0, pwm_in_b=1, inverted=True)
        assert channel_duties(0.4, inverted) == (0.0, 0.4)

    def test_trim_scales_and_result_is_clamped(self) -> None:
        trimmed = MotorChannelConfig(pwm_in_a=0, pwm_in_b=1, trim=1.2)
        assert channel_duties(0.5, trimmed) == pytest.approx((0.6, 0.0))
        assert channel_duties(-1.0, trimmed) == (0.0, 1.0)


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
        assert duties(i2c_bus, motors_config.left) == pytest.approx((0.5, 0.0))
        assert duties(i2c_bus, motors_config.right) == pytest.approx((0.0, 0.25))

    def test_direction_change_lowers_inactive_input_first(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        left = motors_config.left
        driver.set_speeds(TrackSpeeds(left=-0.5, right=0.0))
        i2c_bus.writes.clear()
        driver.set_speeds(TrackSpeeds(left=0.5, right=0.0))
        first_register = i2c_bus.writes[0][1]
        assert first_register == LED0_ON_L + 4 * left.pwm_in_b

    def test_stop_drives_all_inputs_low(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        driver = make_driver(i2c_bus, motors_config, owns_bus=False)
        driver.set_speeds(TrackSpeeds(left=1.0, right=1.0))
        driver.stop()
        assert duties(i2c_bus, motors_config.left) == (0.0, 0.0)
        assert duties(i2c_bus, motors_config.right) == (0.0, 0.0)

    def test_close_stops_and_closes_owned_bus(
        self, i2c_bus: FakeI2CBus, motors_config: MotorsConfig
    ) -> None:
        with make_driver(i2c_bus, motors_config, owns_bus=True) as driver:
            driver.set_speeds(TrackSpeeds(left=0.3, right=0.3))
        assert i2c_bus.closed
        assert duties(i2c_bus, motors_config.left) == (0.0, 0.0)
        assert duties(i2c_bus, motors_config.right) == (0.0, 0.0)

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
