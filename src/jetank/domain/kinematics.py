"""
Differential-drive mixing: body-frame commands to per-track speeds.

Layer: domain
"""

from __future__ import annotations

from jetank.domain.models import DriveCommand, TrackSpeeds


def mix(command: DriveCommand) -> TrackSpeeds:
    """Convert a body-frame command into left/right track speeds.

    Uses ``left = linear - angular`` and ``right = linear + angular``. If either side would
    exceed full scale, both sides are divided by the same factor, which preserves the ratio
    between the tracks (and therefore the turning radius) instead of clipping one side.

    Args:
        command: Normalised linear and angular request.

    Returns:
        Track speeds, each within [-1, 1].
    """
    left = command.linear - command.angular
    right = command.linear + command.angular
    peak = max(abs(left), abs(right))
    if peak > 1.0:
        left /= peak
        right /= peak
    return TrackSpeeds(left=left, right=right)
