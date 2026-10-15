"""In-memory implementations of every interface, for tests and simulation."""

from jetank.drivers.fake.camera import FakeCamera
from jetank.drivers.fake.clock import FakeClock
from jetank.drivers.fake.i2c import FakeI2CBus
from jetank.drivers.fake.motors import FakeMotorDriver

__all__ = ["FakeCamera", "FakeClock", "FakeI2CBus", "FakeMotorDriver"]
