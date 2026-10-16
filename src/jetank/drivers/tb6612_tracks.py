"""
Track motor driver: two TB6612 H-bridge channels whose inputs are fed by a PCA9685.

Layer: drivers

Each motor uses three PCA9685 channels in the Adafruit MotorHAT layout: one carries the PWM
(speed) and two are driven fully high or low to select the direction. Zero speed drives all
three low so the motor coasts. The two direction inputs are never driven high together.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jetank.drivers.pca9685 import Pca9685
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.settings import MotorChannelConfig, MotorsConfig
    from jetank.domain.models import TrackSpeeds
    from jetank.interfaces.i2c import I2CBus

logger = logging.getLogger(__name__)


class BridgeOutputs(BaseModel):
    """Levels for one motor's H-bridge inputs.

    Attributes:
        pwm_duty: Duty cycle of the PWM (speed) input, in [0, 1].
        forward: Whether the forward direction input is high.
        reverse: Whether the reverse direction input is high.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pwm_duty: float = Field(ge=0.0, le=1.0)
    forward: bool
    reverse: bool

    @model_validator(mode="after")
    def _one_direction(self) -> BridgeOutputs:
        if self.forward and self.reverse:
            raise ValueError("forward and reverse inputs must not both be high")
        if self.pwm_duty > 0.0 and not (self.forward or self.reverse):
            raise ValueError("a non-zero duty needs a direction")
        return self


def bridge_outputs(speed: float, channel: MotorChannelConfig) -> BridgeOutputs:
    """H-bridge input levels for one motor after applying trim and inversion.

    Args:
        speed: Normalised speed in [-1, 1]; positive drives the track forward.
        channel: Wiring and calibration of the motor.
    """
    value = speed * channel.trim * (-1.0 if channel.inverted else 1.0)
    value = max(-1.0, min(1.0, value))
    return BridgeOutputs(pwm_duty=abs(value), forward=value > 0.0, reverse=value < 0.0)


class Tb6612TrackDriver:
    """:class:`~jetank.interfaces.MotorDriver` for the JETANK tracks.

    Use :meth:`open` to create one on a real bus. ``bus``, if given, is owned by the driver
    and closed by :meth:`close`; pass ``None`` when the bus is shared and managed elsewhere.
    """

    def __init__(self, pwm: Pca9685, config: MotorsConfig, bus: I2CBus | None = None) -> None:
        self._pwm = pwm
        self._config = config
        self._owned_bus = bus
        self._closed = False
        self._pwm.all_off()

    @classmethod
    def open(cls, config: MotorsConfig) -> Tb6612TrackDriver:
        """Open the I2C bus from ``config`` and initialise the PWM chip.

        Raises:
            HardwareError: The bus cannot be opened or the chip does not respond.
        """
        from smbus2 import SMBus

        try:
            bus = SMBus(config.i2c_bus)
        except OSError as exc:
            raise HardwareError(f"cannot open /dev/i2c-{config.i2c_bus}: {exc}") from exc
        try:
            pwm = Pca9685(bus, config.i2c_address, config.pwm_frequency_hz)
            return cls(pwm, config, bus=bus)
        except BaseException:
            bus.close()
            raise

    def __enter__(self) -> Tb6612TrackDriver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def set_speeds(self, speeds: TrackSpeeds) -> None:
        self._ensure_open()
        self._apply(self._config.left, speeds.left)
        self._apply(self._config.right, speeds.right)

    def stop(self) -> None:
        self._ensure_open()
        self._stop_outputs()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_outputs()
        finally:
            if self._owned_bus is not None:
                self._owned_bus.close()

    def _stop_outputs(self) -> None:
        for channel in (self._config.left, self._config.right):
            self._pwm.set_duty(channel.pwm_channel, 0.0)
            self._pwm.set_duty(channel.forward_channel, 0.0)
            self._pwm.set_duty(channel.reverse_channel, 0.0)

    def _apply(self, channel: MotorChannelConfig, speed: float) -> None:
        outputs = bridge_outputs(speed, channel)
        if outputs.pwm_duty == 0.0:
            self._pwm.set_duty(channel.pwm_channel, 0.0)
        # Lower the inactive direction input before raising the active one, so the two are
        # never high together; the speed is applied once the direction is set.
        if outputs.forward:
            self._pwm.set_duty(channel.reverse_channel, 0.0)
            self._pwm.set_duty(channel.forward_channel, 1.0)
        else:
            self._pwm.set_duty(channel.forward_channel, 0.0)
            self._pwm.set_duty(channel.reverse_channel, 1.0 if outputs.reverse else 0.0)
        if outputs.pwm_duty > 0.0:
            self._pwm.set_duty(channel.pwm_channel, outputs.pwm_duty)

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("track driver is closed")
