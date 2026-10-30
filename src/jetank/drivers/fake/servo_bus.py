"""
In-memory servo bus that simulates SCS servos at the goal/torque level.

Layer: drivers (fake)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jetank.domain.servo import MAX_POSITION, ServoState
from jetank.errors import DriverClosedError, ServoError


@dataclass
class FakeServo:
    """State of one simulated servo. With torque on, it reaches its goal instantly."""

    position: int = 512
    goal: int = 512
    torque_enabled: bool = False
    voltage_v: float = 5.4
    temperature_c: int = 25


@dataclass(frozen=True)
class MoveRecord:
    servo_id: int
    position: int
    duration_s: float
    torque_enabled: bool


@dataclass
class FakeServoBus:
    """:class:`~jetank.interfaces.ServoBus` over simulated servos.

    Attributes:
        servos: Simulated servos by ID; IDs not listed do not answer.
        moves: Every ``move`` call, in order.
        torque_log: Every ``(servo_id, enabled)`` passed to ``set_torque``, in order.
        fail_with: If set, every call except ``close`` raises this.
    """

    servos: dict[int, FakeServo] = field(default_factory=dict)
    moves: list[MoveRecord] = field(default_factory=list)
    torque_log: list[tuple[int, bool]] = field(default_factory=list)
    fail_with: ServoError | None = None
    closed: bool = False

    def ping(self, servo_id: int) -> bool:
        self._check()
        return servo_id in self.servos

    def read_state(self, servo_id: int) -> ServoState:
        servo = self._servo(servo_id)
        return ServoState(
            servo_id=servo_id,
            position=servo.position,
            voltage_v=servo.voltage_v,
            temperature_c=servo.temperature_c,
            moving=False,
            torque_enabled=servo.torque_enabled,
        )

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        servo = self._servo(servo_id)
        self.torque_log.append((servo_id, enabled))
        servo.torque_enabled = enabled
        if enabled:
            servo.position = servo.goal  # a real servo drives to its stored goal

    def move(self, servo_id: int, position: int, duration_s: float) -> None:
        servo = self._servo(servo_id)
        if not 0 <= position <= MAX_POSITION:
            raise ValueError(f"position must be within [0, {MAX_POSITION}], got {position}")
        if duration_s <= 0:
            raise ValueError("duration must be positive")
        self.moves.append(MoveRecord(servo_id, position, duration_s, servo.torque_enabled))
        servo.goal = position
        if servo.torque_enabled:
            servo.position = position

    def close(self) -> None:
        if self.closed:
            return
        for servo_id, servo in self.servos.items():
            if servo.torque_enabled:
                self.torque_log.append((servo_id, False))
                servo.torque_enabled = False
        self.closed = True

    def _check(self) -> None:
        if self.closed:
            raise DriverClosedError("servo bus is closed")
        if self.fail_with is not None:
            raise self.fail_with

    def _servo(self, servo_id: int) -> FakeServo:
        self._check()
        if servo_id not in self.servos:
            raise ServoError(f"servo {servo_id} does not answer")
        return self.servos[servo_id]
