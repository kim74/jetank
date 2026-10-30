"""
Ports for serial bus servos and the byte stream they are reached through.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from jetank.domain.servo import ServoState


@runtime_checkable
class SerialPort(Protocol):
    """Byte stream to the servo bus. ``serial.Serial`` satisfies it structurally."""

    def write(self, data: bytes, /) -> int | None: ...

    def read(self, size: int = 1, /) -> bytes: ...

    def reset_input_buffer(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class ServoBus(Protocol):
    """Servos sharing one serial line, addressed by ID.

    Implementations must make ``close()`` idempotent and switch off the torque of every
    servo they switched on, so no servo is left driving after the bus is released.

    Every method raises :class:`~jetank.errors.HardwareError` (not ``ServoError``) when the
    serial port itself fails, so a dead port is never mistaken for an absent servo.
    """

    def ping(self, servo_id: int) -> bool:
        """Whether a servo with this ID answers."""
        ...

    def read_state(self, servo_id: int) -> ServoState:
        """Read position, voltage, temperature and status.

        Raises:
            ServoError: The servo did not answer or the reply was corrupt.
        """
        ...

    def set_torque(self, servo_id: int, enabled: bool) -> None:
        """Switch the servo's motor on (hold/drive) or off (limp).

        Raises:
            ServoError: The servo did not acknowledge or reported a fault.
        """
        ...

    def move(self, servo_id: int, position: int, duration_s: float) -> None:
        """Set a new goal position, reached in about ``duration_s``. Does not wait.

        With torque off, only the goal is stored and nothing moves.

        Raises:
            ValueError: Position or duration out of range.
            ServoError: The servo did not acknowledge or reported a fault.
        """
        ...

    def close(self) -> None:
        """Switch off torque on servos this bus enabled, then release the port."""
        ...
