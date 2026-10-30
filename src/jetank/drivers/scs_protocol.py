"""
Packet format of Feetech SCS-series serial bus servos (pure encoding, no I/O).

Layer: drivers

Instruction: ``FF FF id len instr params... checksum`` with ``len = len(params) + 2``.
Status reply: ``FF FF id len error params... checksum``. The checksum is the inverted low
byte of the sum of every byte after the two ``FF``. Multi-byte registers are big-endian on
the SCS series (unlike the little-endian STS series).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from jetank.errors import ServoError

HEADER: Final = b"\xff\xff"
BROADCAST_ID: Final = 0xFE

# Instructions
PING: Final = 0x01
READ: Final = 0x02
WRITE: Final = 0x03

# Registers (SCS series memory table)
TORQUE_ENABLE: Final = 40
GOAL_POSITION: Final = 42  # 2 bytes, then goal time (ms, 2 bytes), then speed (2 bytes)
PRESENT_POSITION: Final = 56  # start of the live block read by ``read_state``
PRESENT_VOLTAGE: Final = 62  # 0.1 V units
PRESENT_TEMPERATURE: Final = 63  # degrees Celsius
MOVING: Final = 66
#: Registers PRESENT_POSITION..MOVING read in one transfer.
LIVE_BLOCK_LENGTH: Final = MOVING - PRESENT_POSITION + 1

#: Bytes in a status reply besides its parameters.
STATUS_OVERHEAD: Final = 6


class StatusPacket(BaseModel):
    """A decoded status reply.

    Attributes:
        servo_id: ID of the servo that replied.
        error: Fault bits (0 = no fault).
        params: Payload, e.g. the register bytes of a READ.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    servo_id: int
    error: int
    params: bytes


def checksum(body: bytes) -> int:
    """Inverted low byte of the sum of ``body`` (everything after the header)."""
    return ~sum(body) & 0xFF


def instruction_packet(servo_id: int, instruction: int, params: bytes = b"") -> bytes:
    """Encode one instruction.

    Raises:
        ValueError: ID out of range, or too many parameters for one packet.
    """
    if not 0 <= servo_id <= BROADCAST_ID:
        raise ValueError(f"servo ID must be within [0, {BROADCAST_ID}], got {servo_id}")
    if len(params) > 0xFF - 2:
        raise ValueError("too many parameters for one packet")
    body = bytes([servo_id, len(params) + 2, instruction, *params])
    return HEADER + body + bytes([checksum(body)])


def read_packet(servo_id: int, register: int, length: int) -> bytes:
    """READ ``length`` bytes starting at ``register``."""
    return instruction_packet(servo_id, READ, bytes([register, length]))


def write_packet(servo_id: int, register: int, data: bytes) -> bytes:
    """WRITE ``data`` starting at ``register``."""
    return instruction_packet(servo_id, WRITE, bytes([register, *data]))


def parse_status(data: bytes, expected_id: int, expected_params: int) -> StatusPacket:
    """Decode a status reply.

    Raises:
        ServoError: The reply is truncated, corrupt, from another servo, or has an
            unexpected length.
    """
    expected_size = STATUS_OVERHEAD + expected_params
    if len(data) < expected_size:
        raise ServoError(f"servo {expected_id}: short reply {data.hex(' ') or '(none)'}")
    if data[:2] != HEADER:
        raise ServoError(f"servo {expected_id}: reply without header: {data.hex(' ')}")
    servo_id, length, error = data[2], data[3], data[4]
    if servo_id != expected_id:
        raise ServoError(f"servo {expected_id}: reply came from servo {servo_id}")
    if length != expected_params + 2:
        raise ServoError(
            f"servo {expected_id}: reply length {length}, expected {expected_params + 2}"
        )
    body = data[2 : expected_size - 1]
    if checksum(body) != data[expected_size - 1]:
        raise ServoError(f"servo {expected_id}: checksum mismatch in {data.hex(' ')}")
    return StatusPacket(servo_id=servo_id, error=error, params=bytes(data[5 : expected_size - 1]))


def word(high: int, low: int) -> int:
    """Big-endian 16-bit value."""
    return (high << 8) | low


def word_bytes(value: int) -> bytes:
    """Big-endian bytes of a 16-bit value.

    Raises:
        ValueError: ``value`` does not fit in 16 bits.
    """
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"value must fit in 16 bits, got {value}")
    return bytes([value >> 8, value & 0xFF])
