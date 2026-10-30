"""
Robot status for the on-board display: what is shown, and how it is laid out as text lines.

Layer: domain
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

_VALUE_OBJECT: Final = ConfigDict(frozen=True, extra="forbid")

#: Printable ASCII only: the display font has no other glyphs.
_PRINTABLE: Final = frozenset(chr(code) for code in range(0x20, 0x7F))


class RobotMode(str, Enum):
    """What the robot is currently doing, as shown to the operator."""

    IDLE = "idle"
    TELEOP_WAITING = "teleop: waiting"
    TELEOP_DRIVING = "teleop: driving"


class BatteryState(BaseModel):
    """One battery reading.

    Attributes:
        voltage_v: Battery (bus) voltage in volts.
    """

    model_config = _VALUE_OBJECT

    voltage_v: float = Field(ge=0.0, le=32.0, allow_inf_nan=False)


class RobotStatus(BaseModel):
    """Everything the status display reports.

    Attributes:
        hostname: Network name of the robot.
        ip_address: Address on the network, or ``None`` when not connected.
        battery: Latest battery reading, or ``None`` when the monitor is unavailable.
        mode: Current activity.
    """

    model_config = _VALUE_OBJECT

    hostname: str
    ip_address: str | None
    battery: BatteryState | None
    mode: RobotMode


ScreenLine = Annotated[str, Field(max_length=64)]


class StatusScreen(BaseModel):
    """Text lines for a small monochrome display, top to bottom.

    Attributes:
        lines: Up to 8 lines of printable ASCII. A driver clips lines that do not fit.
    """

    model_config = _VALUE_OBJECT

    lines: tuple[ScreenLine, ...] = Field(max_length=8)

    @field_validator("lines")
    @classmethod
    def _printable(cls, lines: tuple[str, ...]) -> tuple[str, ...]:
        for line in lines:
            bad = set(line) - _PRINTABLE
            if bad:
                raise ValueError(f"line {line!r} has characters the display cannot show")
        return lines


def format_status(
    status: RobotStatus, columns: int, rows: int, low_voltage_v: float
) -> StatusScreen:
    """Lay out ``status`` as at most ``rows`` lines of at most ``columns`` characters.

    Lines in priority order: hostname, IP address, battery, mode. On a display with fewer
    rows the last ones are dropped. Non-printable characters in the hostname become ``?``.

    Args:
        status: What to show.
        columns: Characters per line.
        rows: Number of lines.
        low_voltage_v: At or below this voltage the battery line is flagged ``LOW``.

    Raises:
        ValueError: ``columns`` or ``rows`` is not positive.
    """
    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")
    if status.battery is None:
        battery = "bat --.-V"
    else:
        battery = f"bat {status.battery.voltage_v:4.1f}V"
        if status.battery.voltage_v <= low_voltage_v:
            battery += " LOW"
    hostname = "".join(c if c in _PRINTABLE else "?" for c in status.hostname)
    lines = (
        hostname,
        status.ip_address if status.ip_address is not None else "no network",
        battery,
        status.mode.value,
    )
    return StatusScreen(lines=tuple(line[:columns] for line in lines[:rows]))
