"""Simulated SCS servos behind a serial port, for testing the real bus driver."""

from __future__ import annotations

from dataclasses import dataclass, field

from jetank.drivers import scs_protocol as scs

#: Registers 0..69 read from servo 1 on the JETANK (2026-10-30): position 971, 5.3 V, 22 °C.
JETANK_SERVO_1 = bytes.fromhex(
    "00 05 01 05 0f 01 00 00 01 00 14 03 eb 50 5a 2d 03 e8 00 24 25 0f 0f 00 00 1e 01 01 01 00"
    "00 00 00 00 00 00 00 14 64 50 00 00 03 cb 00 00 00 00 01 00 00 00 00 00 00 00 03 cb 00 00"
    "00 00 35 16 00 00 00 03 cb 00"
)


@dataclass
class FakeScsPort:
    """:class:`~jetank.interfaces.SerialPort` answering like SCS servos.

    Attributes:
        registers: Register bytes per servo ID; IDs not present stay silent.
        errors: Fault bits each servo puts in its replies.
        drop_replies: Number of upcoming replies to swallow (simulated line noise).
        sent: Every instruction packet received, in order.
    """

    registers: dict[int, bytearray] = field(default_factory=dict)
    errors: dict[int, int] = field(default_factory=dict)
    drop_replies: int = 0
    fail_with: OSError | None = None
    sent: list[bytes] = field(default_factory=list)
    closed: bool = False
    _pending: bytes = b""

    def add_servo(self, servo_id: int, image: bytes = JETANK_SERVO_1) -> bytearray:
        regs = bytearray(image) + bytearray(max(0, 70 - len(image)))
        regs[5] = servo_id
        self.registers[servo_id] = regs
        return regs

    def write(self, data: bytes, /) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        self.sent.append(bytes(data))
        servo_id, instruction, params = data[2], data[4], data[5:-1]
        assert data[-1] == scs.checksum(data[2:-1]), "driver sent a bad checksum"
        if servo_id not in self.registers:
            return len(data)
        regs = self.registers[servo_id]
        reply_params = b""
        if instruction == scs.READ:
            reply_params = bytes(regs[params[0] : params[0] + params[1]])
        elif instruction == scs.WRITE:
            regs[params[0] : params[0] + len(params) - 1] = params[1:]
        if self.drop_replies:
            self.drop_replies -= 1
            return len(data)
        body = bytes([servo_id, len(reply_params) + 2, self.errors.get(servo_id, 0)]) + reply_params
        self._pending = scs.HEADER + body + bytes([scs.checksum(body)])
        return len(data)

    def read(self, size: int = 1, /) -> bytes:
        data, self._pending = self._pending[:size], self._pending[size:]
        return data

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def close(self) -> None:
        self.closed = True

    def writes_to(self, servo_id: int, register: int) -> list[bytes]:
        """Data of every WRITE to ``register`` of ``servo_id``, in order."""
        return [
            p[6:-1]
            for p in self.sent
            if p[2] == servo_id and p[4] == scs.WRITE and p[5] == register
        ]
