"""In-memory implementations of every interface, for tests and simulation."""

from jetank.drivers.fake.camera import FakeCamera
from jetank.drivers.fake.clock import FakeClock
from jetank.drivers.fake.display import FakeDisplay
from jetank.drivers.fake.grasp_store import FakeGraspStore
from jetank.drivers.fake.host import FakeHostInfo
from jetank.drivers.fake.i2c import FakeI2CBus
from jetank.drivers.fake.motors import FakeMotorDriver
from jetank.drivers.fake.power import FakePowerMonitor
from jetank.drivers.fake.scene_store import FakeSceneStore
from jetank.drivers.fake.servo_bus import FakeServo, FakeServoBus

__all__ = [
    "FakeCamera",
    "FakeClock",
    "FakeDisplay",
    "FakeGraspStore",
    "FakeHostInfo",
    "FakeI2CBus",
    "FakeMotorDriver",
    "FakePowerMonitor",
    "FakeSceneStore",
    "FakeServo",
    "FakeServoBus",
]
