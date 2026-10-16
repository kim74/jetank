from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest
from websockets.asyncio.client import ClientConnection, connect

from jetank.app.teleop_server import TeleopServer, load_page
from jetank.config.settings import DriveConfig, TeleopConfig
from jetank.control.drive import DriveController
from jetank.control.teleop import TeleopSession
from jetank.domain.models import TrackSpeeds
from jetank.drivers.fake import FakeMotorDriver
from jetank.drivers.system_clock import SystemClock
from jetank.errors import HardwareError, ServerError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

PAGE = b"<!doctype html><title>test page</title>"
CONFIG = TeleopConfig(host="127.0.0.1", port=0, tick_period_s=0.01)
WATCHDOG_TIMEOUT_S = 0.1
FORWARD = json.dumps({"type": "drive", "linear": 1.0, "angular": 0.0})


class FailingMotors(FakeMotorDriver):
    def set_speeds(self, speeds: TrackSpeeds) -> None:
        raise HardwareError("bus gone")


def make_server(motors: FakeMotorDriver) -> TeleopServer:
    drive_config = DriveConfig(max_speed=0.5, watchdog_timeout_s=WATCHDOG_TIMEOUT_S)
    drive = DriveController(motors, SystemClock(), drive_config)
    return TeleopServer(TeleopSession(drive), CONFIG, PAGE)


def run_with_server(motors: FakeMotorDriver, body: Callable[[int], Awaitable[None]]) -> None:
    """Start a real server on a free local port, run ``body(port)``, then shut it down."""

    async def main() -> None:
        server = make_server(motors)
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(server.serve(on_ready=ready.set_result))
        port = await asyncio.wait_for(ready, 5)
        try:
            await body(port)
        finally:
            server.request_stop()
            await asyncio.wait_for(task, 5)

    asyncio.run(main())


async def receive(ws: ClientConnection) -> dict[str, Any]:
    status: dict[str, Any] = json.loads(await asyncio.wait_for(ws.recv(), 5))
    return status


async def eventually(condition: Callable[[], bool], timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


def test_serves_the_control_page() -> None:
    async def body(port: int) -> None:
        url = f"http://127.0.0.1:{port}/"
        with await asyncio.to_thread(urllib.request.urlopen, url) as response:
            assert response.status == 200
            assert response.headers["Content-Type"].startswith("text/html")
            assert response.read() == PAGE

    run_with_server(FakeMotorDriver(), body)


def test_unknown_path_is_404() -> None:
    async def body(port: int) -> None:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            await asyncio.to_thread(urllib.request.urlopen, f"http://127.0.0.1:{port}/nope")
        excinfo.value.close()
        assert excinfo.value.code == 404

    run_with_server(FakeMotorDriver(), body)


def test_driver_drives_and_disconnect_stops() -> None:
    motors = FakeMotorDriver()

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            assert (await receive(ws))["role"] == "driver"
            await ws.send(FORWARD)
            status = await receive(ws)
            assert status["moving"] is True
            assert status["speeds"] == {"left": 0.5, "right": 0.5}
            assert motors.current == TrackSpeeds(left=0.5, right=0.5)
        await eventually(lambda: motors.current.is_stopped)

    run_with_server(motors, body)


def test_second_client_observes_but_can_stop() -> None:
    motors = FakeMotorDriver()

    async def body(port: int) -> None:
        url = f"ws://127.0.0.1:{port}/ws"
        async with connect(url) as driver, connect(url) as observer:
            await receive(driver)
            assert (await receive(observer))["role"] == "observer"
            await driver.send(FORWARD)
            await receive(driver)
            await observer.send(FORWARD)
            assert (await receive(observer))["error"] == "another client is driving"
            await observer.send(json.dumps({"type": "stop"}))
            assert (await receive(observer))["moving"] is False
            assert motors.current.is_stopped

    run_with_server(motors, body)


def test_watchdog_stops_a_silent_driver() -> None:
    motors = FakeMotorDriver()

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await receive(ws)
            await ws.send(FORWARD)
            await receive(ws)
            assert not motors.current.is_stopped
            await asyncio.sleep(WATCHDOG_TIMEOUT_S * 3)
            assert motors.current.is_stopped

    run_with_server(motors, body)


def test_invalid_message_from_driver_stops() -> None:
    motors = FakeMotorDriver()

    async def body(port: int) -> None:
        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await receive(ws)
            await ws.send(FORWARD)
            await receive(ws)
            await ws.send("{not json")
            status = await receive(ws)
            assert status["error"] == "invalid message"
            assert motors.current.is_stopped

    run_with_server(motors, body)


def test_motor_failure_shuts_the_server_down() -> None:
    motors = FailingMotors()

    async def main() -> None:
        server = make_server(motors)
        ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(server.serve(on_ready=ready.set_result))
        port = await asyncio.wait_for(ready, 5)
        async with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await receive(ws)
            await ws.send(FORWARD)
            with pytest.raises(HardwareError, match="bus gone"):
                await asyncio.wait_for(task, 5)

    asyncio.run(main())
    assert motors.current.is_stopped


def test_request_stop_before_serve_is_harmless() -> None:
    make_server(FakeMotorDriver()).request_stop()


def test_packaged_page_is_the_remote_control() -> None:
    page = load_page().decode("utf-8")
    assert "<title>JETANK Remote</title>" in page
    assert "/ws" in page


def test_port_in_use_raises_server_error() -> None:
    async def body(port: int) -> None:
        taken = TeleopConfig(host="127.0.0.1", port=port, tick_period_s=0.01)
        drive = DriveController(FakeMotorDriver(), SystemClock(), DriveConfig())
        second = TeleopServer(TeleopSession(drive), taken, PAGE)
        with pytest.raises(ServerError, match=f"127.0.0.1:{port}"):
            await asyncio.wait_for(second.serve(), 5)

    run_with_server(FakeMotorDriver(), body)
