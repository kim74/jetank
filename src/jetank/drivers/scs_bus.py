"""
Feetech SCS-series servo bus over a serial port (the JETANK's ``/dev/ttyTHS1``).

Layer: drivers

Every transfer is request/reply: one instruction packet, one status reply. A missing or
corrupt reply is retried a configured number of times, because single replies are
occasionally lost on the half-duplex line.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Final

from jetank.domain.servo import MAX_POSITION, ServoState
from jetank.drivers import scs_protocol as scs
from jetank.errors import DriverClosedError, HardwareError, ServoError

if TYPE_CHECKING:
    from types import TracebackType

    from jetank.config.servos import ServosConfig
    from jetank.interfaces.servo import SerialPort

logger = logging.getLogger(__name__)

#: Longest move time the goal-time register can hold, in seconds.
MAX_DURATION_S: Final = 0xFFFF / 1000


class ScsServoBus:
    """:class:`~jetank.interfaces.ServoBus` speaking the SCS protocol.

    Use :meth:`open` to create one on a real port. The bus owns ``port`` and closes it.

    Args:
        port: Serial port whose ``read`` returns after the reply timeout at the latest.
        retries: Extra attempts after a missing or corrupt reply.
    """

    def __init__(self, port: SerialPort, retries: int = 2) -> None:
        self._port = port
        self._retries = retries
        self._torque_on: set[int] = set()
        self._closed = False

    @classmethod
    def open(cls, config: ServosConfig) -> ScsServoBus:
        """Open the serial port from ``config``.

        Raises:
            HardwareError: The port cannot be opened (missing, or no ``dialout`` access).
        """
        import serial

        try:
            port = serial.Serial(config.port, config.baud_rate, timeout=config.response_timeout_s)
        except (OSError, serial.SerialException) as exc:
            raise HardwareError(f"cannot open servo bus {config.port}: {exc}") from exc
        logger.info("servo bus %s opened at %d baud", config.port, config.baud_rate)
        return cls(port, config.retries)

    def __enter__(self) -> ScsServoBus:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def ping(self, servo_id: int) -> bool:
        try:
            self._transact(scs.instruction_packet(servo_id, scs.PING), servo_id, 0)
        except ServoError:
            return False
        return True

    def read_state(self, servo_id: int) -> ServoState:
        torque = self._read(servo_id, scs.TORQUE_ENABLE, 1)
        live = self._read(servo_id, scs.PRESENT_POSITION, scs.LIVE_BLOCK_LENGTH)
        at = scs.PRESENT_POSITION
        return ServoState(
            servo_id=servo_id,
            position=min(MAX_POSITION, scs.word(live.params[0], live.params[1])),
            voltage_v=live.params[scs.PRESENT_VOLTAGE - at] / 10,
            temperature_c=live.params[scs.PRESENT_TEMPERATURE - at],
            moving=bool(live.params[scs.MOVING - at]),
            torque_enabled=bool(torque.params[0]),
            error_flags=live.error,
        )

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        if enabled:
            # Tracked before the reply: a lost acknowledgement may hide a servo that did
            # switch on, and close() must still switch it off.
            self._torque_on.add(servo_id)
        self._write(servo_id, scs.TORQUE_ENABLE, bytes([int(enabled)]))
        if not enabled:
            self._torque_on.discard(servo_id)

    def move(self, servo_id: int, position: int, duration_s: float) -> None:
        """Set a goal reached in about ``duration_s``.

        Both the goal time and a matching speed (counts per second) are sent, so the move
        stays slow whichever of the two the servo firmware honours.
        """
        if not 0 <= position <= MAX_POSITION:
            raise ValueError(f"position must be within [0, {MAX_POSITION}], got {position}")
        if not 0.0 < duration_s <= MAX_DURATION_S:
            raise ValueError(f"duration must be within (0, {MAX_DURATION_S}] s, got {duration_s}")
        present = self.read_state(servo_id).position
        speed = max(1, math.ceil(abs(position - present) / duration_s))
        data = (
            scs.word_bytes(position)
            + scs.word_bytes(round(duration_s * 1000))
            + scs.word_bytes(min(speed, 0xFFFF))
        )
        # Some firmware switches torque on when a goal arrives, so close() must switch it off.
        self._torque_on.add(servo_id)
        self._write(servo_id, scs.GOAL_POSITION, data)

    def close(self) -> None:
        if self._closed:
            return
        try:
            for servo_id in sorted(self._torque_on):
                try:
                    self.set_torque(servo_id, False)
                except ServoError as exc:
                    logger.error("could not switch off torque of servo %d: %s", servo_id, exc)
        finally:
            self._closed = True
            self._port.close()

    def _read(self, servo_id: int, register: int, length: int) -> scs.StatusPacket:
        return self._transact(scs.read_packet(servo_id, register, length), servo_id, length)

    def _write(self, servo_id: int, register: int, data: bytes) -> None:
        reply = self._transact(scs.write_packet(servo_id, register, data), servo_id, 0)
        if reply.error:
            raise ServoError(f"servo {servo_id} reported fault 0x{reply.error:02X}")

    def _transact(self, packet: bytes, servo_id: int, reply_params: int) -> scs.StatusPacket:
        if self._closed:
            raise DriverClosedError("servo bus is closed")
        expected = scs.STATUS_OVERHEAD + reply_params
        last_error: ServoError | None = None
        for _attempt in range(self._retries + 1):
            try:
                self._port.reset_input_buffer()
                self._port.write(packet)
                reply = self._port.read(expected)
            except OSError as exc:
                # Not a ServoError: the port itself failed, not one servo's reply.
                raise HardwareError(f"servo bus I/O failed: {exc}") from exc
            try:
                return scs.parse_status(reply, servo_id, reply_params)
            except ServoError as exc:
                last_error = exc
                logger.debug("retrying after bad reply: %s", exc)
        assert last_error is not None
        raise last_error
