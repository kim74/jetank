from __future__ import annotations

import pytest

from jetank.drivers import scs_protocol as scs
from jetank.drivers.scs_bus import ScsServoBus
from jetank.errors import DriverClosedError, HardwareError, ServoError
from tests.unit.drivers.scs_helpers import FakeScsPort


@pytest.fixture
def port() -> FakeScsPort:
    port = FakeScsPort()
    port.add_servo(1)
    port.add_servo(4)
    return port


@pytest.fixture
def bus(port: FakeScsPort) -> ScsServoBus:
    return ScsServoBus(port, retries=2)


def test_ping_finds_present_servos_only(bus: ScsServoBus) -> None:
    assert bus.ping(1)
    assert not bus.ping(2)


def test_read_state_decodes_jetank_registers(bus: ScsServoBus) -> None:
    state = bus.read_state(1)
    assert state.position == 971
    assert state.voltage_v == pytest.approx(5.3)
    assert state.temperature_c == 22
    assert not state.torque_enabled
    assert not state.moving
    assert state.error_flags == 0


def test_read_state_reports_fault_bits(bus: ScsServoBus, port: FakeScsPort) -> None:
    port.errors[1] = 0x20  # overload
    assert bus.read_state(1).error_flags == 0x20


def test_set_torque_writes_enable_register(bus: ScsServoBus, port: FakeScsPort) -> None:
    bus.set_torque(4, True)
    bus.set_torque(4, False)
    assert port.writes_to(4, scs.TORQUE_ENABLE) == [b"\x01", b"\x00"]


def test_move_sends_goal_time_and_matching_speed(bus: ScsServoBus, port: FakeScsPort) -> None:
    # Servo rests at 971; 200 counts in 0.5 s is 400 counts/s.
    bus.move(1, 771, 0.5)
    assert port.writes_to(1, scs.GOAL_POSITION) == [
        scs.word_bytes(771) + scs.word_bytes(500) + scs.word_bytes(400)
    ]


def test_move_to_present_position_uses_minimum_speed(bus: ScsServoBus, port: FakeScsPort) -> None:
    bus.move(1, 971, 0.1)
    assert port.writes_to(1, scs.GOAL_POSITION)[0][4:] == scs.word_bytes(1)


@pytest.mark.parametrize(
    ("position", "duration_s"), [(-1, 1.0), (1024, 1.0), (500, 0.0), (500, 70.0)]
)
def test_move_rejects_out_of_range_values(
    bus: ScsServoBus, port: FakeScsPort, position: int, duration_s: float
) -> None:
    with pytest.raises(ValueError, match="must be within"):
        bus.move(1, position, duration_s)
    assert port.sent == []


def test_lost_reply_is_retried(bus: ScsServoBus, port: FakeScsPort) -> None:
    port.drop_replies = 2
    assert bus.read_state(1).position == 971


def test_gives_up_after_retries(bus: ScsServoBus, port: FakeScsPort) -> None:
    port.drop_replies = 3
    with pytest.raises(ServoError, match="short reply"):
        bus.read_state(4)


def test_silent_servo_raises(bus: ScsServoBus) -> None:
    with pytest.raises(ServoError):
        bus.read_state(2)


def test_write_fault_raises(bus: ScsServoBus, port: FakeScsPort) -> None:
    port.errors[4] = 0x02
    with pytest.raises(ServoError, match="fault 0x02"):
        bus.set_torque(4, False)


def test_port_failure_is_not_mistaken_for_an_absent_servo(
    bus: ScsServoBus, port: FakeScsPort
) -> None:
    port.fail_with = OSError(5, "Input/output error")
    with pytest.raises(HardwareError, match="I/O failed") as raised:
        bus.ping(1)
    assert not isinstance(raised.value, ServoError)


def test_close_switches_off_torque_it_enabled_and_closes_port(
    bus: ScsServoBus, port: FakeScsPort
) -> None:
    bus.set_torque(4, True)
    with bus:
        pass
    assert port.writes_to(4, scs.TORQUE_ENABLE) == [b"\x01", b"\x00"]
    assert port.writes_to(1, scs.TORQUE_ENABLE) == []
    assert port.closed
    bus.close()  # idempotent
    with pytest.raises(DriverClosedError):
        bus.ping(1)


def test_torque_counts_as_on_even_if_the_enable_reply_is_lost(
    port: FakeScsPort,
) -> None:
    bus = ScsServoBus(port, retries=0)
    port.drop_replies = 1
    with pytest.raises(ServoError):
        bus.set_torque(4, True)
    bus.close()
    assert port.writes_to(4, scs.TORQUE_ENABLE) == [b"\x01", b"\x00"]


def test_close_still_closes_port_if_torque_off_fails(bus: ScsServoBus, port: FakeScsPort) -> None:
    bus.set_torque(4, True)
    port.errors[4] = 0x02
    bus.close()
    assert port.closed


def test_close_switches_off_torque_after_a_goal_write(bus: ScsServoBus, port: FakeScsPort) -> None:
    bus.move(4, 900, 0.5)  # some firmware enables torque on a goal write
    bus.close()
    assert port.writes_to(4, scs.TORQUE_ENABLE) == [b"\x00"]


def test_torque_limit_is_written_as_tenths_of_a_percent(
    bus: ScsServoBus, port: FakeScsPort
) -> None:
    bus.set_torque_limit(4, 0.5)
    assert port.writes_to(4, scs.MAX_TORQUE) == [scs.word_bytes(500)]


def test_torque_limit_reads_back(bus: ScsServoBus) -> None:
    assert bus.read_torque_limit(4) == 1.0
    bus.set_torque_limit(4, 0.35)
    assert bus.read_torque_limit(4) == pytest.approx(0.35)


def test_unchanged_torque_limit_is_not_rewritten(bus: ScsServoBus, port: FakeScsPort) -> None:
    bus.set_torque_limit(4, 1.0)  # the servo already holds 1000
    bus.set_torque_limit(4, 0.5)
    bus.set_torque_limit(4, 0.5)
    assert port.writes_to(4, scs.MAX_TORQUE) == [scs.word_bytes(500)]


@pytest.mark.parametrize("fraction", [0.0, 1.01, -0.5])
def test_torque_limit_out_of_range_is_rejected(
    bus: ScsServoBus, port: FakeScsPort, fraction: float
) -> None:
    with pytest.raises(ValueError, match="torque limit"):
        bus.set_torque_limit(4, fraction)
    assert port.sent == []


@pytest.mark.parametrize(
    ("raw", "load"), [(0x0000, 0.0), (0x01F4, 0.5), (0x05F4, -0.5), (0x03FF, 1.0)]
)
def test_load_is_decoded_with_direction(
    bus: ScsServoBus, port: FakeScsPort, raw: int, load: float
) -> None:
    port.registers[4][scs.PRESENT_LOAD : scs.PRESENT_LOAD + 2] = scs.word_bytes(raw)
    assert bus.read_state(4).load == pytest.approx(load)
