"""
Scene recording: saves camera images of the floors with the operator's labels, for training.

Layer: control

Nothing moves here: the recorder takes the newest camera image, reads where the arm is (so
the camera's viewpoint is known), and hands both to a :class:`~jetank.interfaces.SceneStore`.
Automatic recording takes one scene right away and then one every ``auto_period_s`` on
:meth:`SceneRecorder.tick`, until it is stopped, a recording fails, or ``auto_limit_s`` has
passed, so it can never run on unattended.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.domain.records import record_id_for
from jetank.domain.scenes import SceneAction, SceneRecord, SceneReport
from jetank.domain.servo import ArmPose
from jetank.errors import CameraError, HardwareError, ServoError, StorageError

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from jetank.control.arm import ArmController
    from jetank.domain.scenes import SceneLabels, SceneMessage
    from jetank.interfaces.clock import Clock
    from jetank.interfaces.scene_store import SceneStore

logger = logging.getLogger(__name__)


class SceneRecorder:
    """Records labelled scenes, one at a time or automatically. The arm is borrowed.

    Not thread-safe: call it from the thread that owns the arm.

    Args:
        store: Where scenes are saved.
        snapshot: Returns the newest camera image as JPEG, or ``None`` if there is none yet.
            May raise :class:`~jetank.errors.CameraError`.
        arm: Arm whose joints are read with every scene, or ``None`` without an arm.
        now: Wall-clock time with time zone (injected for tests).
        clock: Monotonic time for the automatic recording.
        auto_period_s: Seconds between automatic recordings.
        auto_limit_s: Automatic recording stops by itself after this many seconds.

    Raises:
        ValueError: ``auto_period_s`` or ``auto_limit_s`` is not positive.
    """

    def __init__(
        self,
        store: SceneStore,
        snapshot: Callable[[], bytes | None],
        arm: ArmController | None,
        *,
        now: Callable[[], datetime],
        clock: Clock,
        auto_period_s: float,
        auto_limit_s: float,
    ) -> None:
        if auto_period_s <= 0 or auto_limit_s <= 0:
            raise ValueError("automatic recording period and limit must be positive")
        self._store = store
        self._snapshot = snapshot
        self._arm = arm
        self._now = now
        self._clock = clock
        self._auto_period_s = auto_period_s
        self._auto_limit_s = auto_limit_s
        self._auto_labels: SceneLabels | None = None
        self._next_at_s = 0.0
        self._stop_at_s = 0.0
        self._taken = 0
        self._last: SceneRecord | None = None
        self._error: str | None = None

    @property
    def auto(self) -> bool:
        """Whether scenes are being recorded automatically."""
        return self._auto_labels is not None

    def handle(self, message: SceneMessage) -> str | None:
        """Carry out a scene request. Returns an error text instead of raising."""
        if message.action is SceneAction.AUTO_STOP:
            self.stop_auto()
            return None
        assert message.labels is not None  # SceneMessage guarantees labels unless stopping
        if message.action is SceneAction.TAKE:
            return self._take_reporting(message.labels)
        error = self._take_reporting(message.labels)
        if error is None:
            now_s = self._clock.monotonic()
            self._auto_labels = message.labels
            self._next_at_s = now_s + self._auto_period_s
            self._stop_at_s = now_s + self._auto_limit_s
            logger.info(
                "scenes: recording every %.1f s for at most %.0f s",
                self._auto_period_s,
                self._auto_limit_s,
            )
        return error

    def stop_auto(self) -> None:
        """Stop automatic recording, if it runs. Idempotent."""
        if self._auto_labels is not None:
            logger.info("scenes: automatic recording stopped after %d scenes", self._taken)
        self._auto_labels = None

    def tick(self) -> None:
        """Record the next automatic scene when it is due. Never raises.

        A failed recording stops the automatic recording; the error shows in :meth:`report`.
        """
        if self._auto_labels is None:
            return
        now_s = self._clock.monotonic()
        if now_s >= self._stop_at_s:
            logger.info("scenes: time limit of %.0f s reached", self._auto_limit_s)
            self.stop_auto()
            return
        if now_s < self._next_at_s:
            return
        self._next_at_s = now_s + self._auto_period_s
        if self._take_reporting(self._auto_labels) is not None:
            self.stop_auto()

    def take(self, labels: SceneLabels) -> SceneRecord:
        """Save the newest camera image with ``labels`` and the arm's pose.

        An arm that does not answer does not stop the recording: the scene is saved without
        a pose (and a warning is logged).

        Returns:
            The scene as stored.

        Raises:
            CameraError: There is no camera image.
            StorageError: The scene could not be saved.
        """
        jpeg = self._snapshot()
        if jpeg is None:
            raise CameraError("no camera image (is the live video showing?)")
        moment = self._now()
        record = SceneRecord(
            record_id=record_id_for(moment),
            recorded_at=moment,
            labels=labels,
            arm_pose=self._read_arm_pose(),
        )
        saved = self._store.save(record, jpeg)
        self._taken += 1
        self._last = saved
        return saved

    def report(self) -> SceneReport:
        """Scene recording as shown to the operator."""
        remaining_s = None
        if self._auto_labels is not None:
            remaining_s = max(0.0, self._stop_at_s - self._clock.monotonic())
        return SceneReport(
            auto=self.auto,
            auto_remaining_s=remaining_s,
            taken=self._taken,
            last_record_id=None if self._last is None else self._last.record_id,
            error=self._error,
        )

    def _take_reporting(self, labels: SceneLabels) -> str | None:
        try:
            record = self.take(labels)
        except (HardwareError, StorageError) as exc:
            logger.warning("scenes: could not record a scene: %s", exc)
            self._error = f"scene not recorded: {exc}"
            return self._error
        logger.info(
            "scenes: %s in %s (%s, %s)",
            record.record_id,
            labels.room,
            labels.lighting.value,
            "animals" if labels.has_animals else "no animals",
        )
        self._error = None
        return None

    def _read_arm_pose(self) -> ArmPose | None:
        if self._arm is None or not self._arm.joints:
            return None
        try:
            states = self._arm.read_pose()
        except ServoError as exc:
            logger.warning("scenes: arm not read, scene saved without its pose: %s", exc)
            return None
        return ArmPose(positions={joint: state.position for joint, state in states.items()})
