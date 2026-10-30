"""In-memory implementations of every interface, for tests and simulation."""

from jetank.drivers.fake.camera import FakeCamera
from jetank.drivers.fake.clock import FakeClock
from jetank.drivers.fake.display import FakeDisplay
from jetank.drivers.fake.host import FakeHostInfo
from jetank.drivers.fake.i2c import FakeI2CBus
from jetank.drivers.fake.motors import FakeMotorDriver
from jetank.drivers.fake.power import FakePowerMonitor

__all__ = [
    "FakeCamera",
    "FakeClock",
    "FakeDisplay",
    "FakeHostInfo",
    "FakeI2CBus",
    "FakeMotorDriver",
    "FakePowerMonitor",
]
