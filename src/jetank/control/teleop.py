"""
Remote-control session: decides which connected client may drive and forwards its commands.

Layer: control

Network-agnostic: the server calls :meth:`TeleopSession.connect`, :meth:`~TeleopSession.handle`
and :meth:`~TeleopSession.disconnect` per client, and :meth:`~TeleopSession.tick` periodically.
Safety rules:

* Only the first client to connect drives; later clients are observers until it leaves.
* Any client may stop the robot.
* The driver disconnecting, or sending a malformed message, stops the robot.
* If the driver goes quiet, the :class:`~jetank.control.drive.DriveController` watchdog stops
  the robot on the next :meth:`~TeleopSession.tick`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.domain.models import TrackSpeeds
from jetank.domain.teleop import DriveMessage, Role, StopMessage, TeleopStatus
from jetank.errors import ProtocolError

if TYPE_CHECKING:
    from jetank.control.drive import DriveController

logger = logging.getLogger(__name__)


class TeleopSession:
    """Arbitrates remote clients in front of one :class:`DriveController`.

    Not thread-safe: call it from a single thread or event loop. The controller is borrowed,
    not owned; the caller stops and closes the hardware.
    """

    def __init__(self, drive: DriveController) -> None:
        self._drive = drive
        self._clients: set[str] = set()
        self._driver: str | None = None
        self._speeds = TrackSpeeds.stopped()

    @property
    def driver(self) -> str | None:
        """ID of the client currently allowed to drive, if any."""
        return self._driver

    def connect(self, client_id: str) -> TeleopStatus:
        """Register a client. It becomes the driver if nobody else is driving.

        Raises:
            ProtocolError: A client with this ID is already connected.
        """
        if client_id in self._clients:
            raise ProtocolError(f"client {client_id} is already connected")
        self._clients.add(client_id)
        if self._driver is None:
            self._driver = client_id
            logger.info("teleop: %s is now driving", client_id)
        else:
            logger.info("teleop: %s connected as observer (%s is driving)", client_id, self._driver)
        return self.status(client_id)

    def disconnect(self, client_id: str) -> None:
        """Forget a client. If it was driving, the robot stops and the driver seat is free."""
        self._clients.discard(client_id)
        if client_id == self._driver:
            self._driver = None
            self._stop()
            logger.info("teleop: driver %s disconnected, robot stopped", client_id)

    def handle(self, client_id: str, message: DriveMessage | StopMessage) -> TeleopStatus:
        """Apply one message from a client and report the resulting state.

        Raises:
            ProtocolError: The client is not connected.
            HardwareError: The motors failed (the controller has already tried to stop them).
        """
        self._require_connected(client_id)
        if isinstance(message, StopMessage):
            self._stop()
            return self.status(client_id)
        if client_id != self._driver:
            return self.status(client_id, error="another client is driving")
        self._speeds = self._drive.command(message.to_command())
        return self.status(client_id)

    def reject(self, client_id: str, error: str) -> TeleopStatus:
        """Report an unusable message. If it came from the driver, stop to be safe.

        Raises:
            ProtocolError: The client is not connected.
        """
        self._require_connected(client_id)
        if client_id == self._driver:
            self._stop()
        return self.status(client_id, error=error)

    def tick(self) -> bool:
        """Run the drive watchdog. Returns True if it stopped the robot just now."""
        return self._drive.tick()

    def status(self, client_id: str, error: str | None = None) -> TeleopStatus:
        """Current state as seen by ``client_id``."""
        moving = self._drive.is_moving
        return TeleopStatus(
            role=Role.DRIVER if client_id == self._driver else Role.OBSERVER,
            moving=moving,
            speeds=self._speeds if moving else TrackSpeeds.stopped(),
            error=error,
        )

    def _stop(self) -> None:
        self._drive.stop()
        self._speeds = TrackSpeeds.stopped()

    def _require_connected(self, client_id: str) -> None:
        if client_id not in self._clients:
            raise ProtocolError(f"client {client_id} is not connected")
