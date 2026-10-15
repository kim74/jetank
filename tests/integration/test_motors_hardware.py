"""Real track motors. Needs `pytest -m hardware` AND JETANK_ALLOW_MOTION=1.

Lift the tracks off the ground before running.
"""

from __future__ import annotations

import os
import time

import pytest

from jetank.config.settings import load_settings
from jetank.domain.models import TrackSpeeds
from jetank.drivers.tb6612_tracks import Tb6612TrackDriver

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("JETANK_ALLOW_MOTION") != "1",
        reason="set JETANK_ALLOW_MOTION=1 to allow the motors to move",
    ),
]


def test_tracks_pulse_forward_then_stop() -> None:
    config = load_settings().motors
    with Tb6612TrackDriver.open(config) as driver:
        driver.set_speeds(TrackSpeeds(left=0.3, right=0.3))
        time.sleep(0.5)
        driver.stop()
