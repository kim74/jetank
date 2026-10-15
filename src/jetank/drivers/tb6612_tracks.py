"""
Track motor driver: two TB6612 H-bridge channels whose inputs are fed by a PCA9685.

Layer: drivers

Each motor uses two PWM channels. Forward puts the PWM on IN_A with IN_B low; reverse
does the opposite; both low lets the motor coast. This matches the Waveshare JetBot board
logic in the legacy ``jetbot/motor.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.drivers.pca9685 import Pca9685
from jetank.errors import DriverClosedError, HardwareError

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.settings import MotorChannelConfig, MotorsConfig
    from jetank.domain.models import TrackSpeeds
    from jetank.interfaces.i2c import I2CBus

logger = logging.getLogger(__name__)


def channel_duties(speed: float, channel: MotorChannelConfig) -> tuple[float, float]:
    """Duty cycles ``(in_a, in_b)`` for one motor after applying trim and inversion.

    Args:
        speed: Normalised speed in [-1, 1].
        channel: Wiring and calibration of the motor.
    """
    value = speed * channel.trim * (-1.0 if channel.inverted else 1.0)
    value = max(-1.0, min(1.0, value))
    return (value, 0.0) if value >= 0.0 else (0.0, -value)


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
            self._pwm.set_duty(channel.pwm_in_a, 0.0)
            self._pwm.set_duty(channel.pwm_in_b, 0.0)

    def _apply(self, channel: MotorChannelConfig, speed: float) -> None:
        duty_a, duty_b = channel_duties(speed, channel)
        # Lower the inactive input first so the bridge never sees both inputs driven.
        if duty_a > 0.0:
            self._pwm.set_duty(channel.pwm_in_b, duty_b)
            self._pwm.set_duty(channel.pwm_in_a, duty_a)
        else:
            self._pwm.set_duty(channel.pwm_in_a, duty_a)
            self._pwm.set_duty(channel.pwm_in_b, duty_b)

    def _ensure_open(self) -> None:
        if self._closed:
            raise DriverClosedError("track driver is closed")
