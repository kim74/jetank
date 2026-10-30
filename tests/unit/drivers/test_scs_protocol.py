from __future__ import annotations

import pytest

from jetank.drivers import scs_protocol as scs
from jetank.errors import ServoError


def test_ping_packet_matches_feetech_example() -> None:
    assert scs.instruction_packet(1, scs.PING) == bytes.fromhex("ff ff 01 02 01 fb")


def test_read_and_write_packets() -> None:
    assert scs.read_packet(4, 56, 2) == bytes.fromhex("ff ff 04 04 02 38 02 bb")
    packet = scs.write_packet(4, 40, b"\x01")
    assert packet[:6] == bytes.fromhex("ff ff 04 04 03 28")
    assert packet[-1] == scs.checksum(packet[2:-1])


@pytest.mark.parametrize("servo_id", [-1, 255])
def test_packet_rejects_bad_id(servo_id: int) -> None:
    with pytest.raises(ValueError, match="servo ID"):
        scs.instruction_packet(servo_id, scs.PING)


def test_packet_rejects_oversized_payload() -> None:
    with pytest.raises(ValueError, match="too many"):
        scs.instruction_packet(1, scs.WRITE, bytes(254))


def status(servo_id: int, error: int, params: bytes) -> bytes:
    body = bytes([servo_id, len(params) + 2, error]) + params
    return scs.HEADER + body + bytes([scs.checksum(body)])


def test_parse_valid_reply() -> None:
    reply = scs.parse_status(status(4, 0x20, b"\x03\xcf"), 4, 2)
    assert (reply.servo_id, reply.error, reply.params) == (4, 0x20, b"\x03\xcf")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "short reply"),
        (b"\x00\xff" + status(4, 0, b"\x00\x00")[2:], "without header"),
        (status(3, 0, b"\x00\x00"), "came from servo 3"),
        (status(4, 0, b"\x00\x00\x00"), "reply length"),
        (status(4, 0, b"\x00\x00")[:-1] + b"\x00", "checksum"),
    ],
)
def test_parse_rejects_bad_replies(data: bytes, message: str) -> None:
    with pytest.raises(ServoError, match=message):
        scs.parse_status(data, 4, 2)


def test_words_are_big_endian() -> None:
    assert scs.word(0x03, 0xCB) == 971
    assert scs.word_bytes(971) == b"\x03\xcb"
    with pytest.raises(ValueError, match="16 bits"):
        scs.word_bytes(0x10000)
