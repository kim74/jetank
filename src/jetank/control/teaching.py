"""
Grasp teaching: records the arm's present pose together with the operator's labels.

Layer: control

Nothing moves here: the teacher only reads the joints and the gripper, grabs the current
camera image, and hands the record to a :class:`~jetank.interfaces.GraspStore`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from jetank.domain.teaching import GraspRecord, GripperReading, record_id_for

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from jetank.control.arm import ArmController
    from jetank.control.gripper import GripperController
    from jetank.domain.teaching import TeachLabels
    from jetank.interfaces.grasp_store import GraspStore

logger = logging.getLogger(__name__)


class GraspTeacher:
    """Takes labelled snapshots of the arm. Controllers and store are borrowed, not owned.

    Args:
        arm: Arm whose joints are read.
        gripper: Gripper, or ``None`` if the robot has none.
        store: Where records are saved.
        snapshot: Returns the newest camera image as JPEG, or ``None`` if there is none.
        now: Wall-clock time with time zone (injected for tests).
        holding_margin: Counts short of the grip position at which a closed gripper counts as
            holding something; see :meth:`~jetank.domain.servo.GripperCalibration.is_holding`.

    Raises:
        ValueError: ``holding_margin`` is not positive.
    """

    def __init__(
        self,
        arm: ArmController,
        gripper: GripperController | None,
        store: GraspStore,
        snapshot: Callable[[], bytes | None],
        now: Callable[[], datetime],
        holding_margin: int,
    ) -> None:
        if holding_margin < 1:
            raise ValueError("holding margin must be positive")
        self._arm = arm
        self._gripper = gripper
        self._store = store
        self._snapshot = snapshot
        self._now = now
        self._holding_margin = holding_margin

    def take(self, labels: TeachLabels) -> GraspRecord:
        """Read the arm and gripper, and save them with ``labels`` and the camera image.

        Returns:
            The record as stored.

        Raises:
            ServoError: A servo did not answer; nothing is saved.
            StorageError: The record could not be saved.
        """
        joints = self._arm.read_pose()
        gripper = None if self._gripper is None else self._read_gripper()
        moment = self._now()
        record = GraspRecord(
            record_id=record_id_for(moment),
            recorded_at=moment,
            labels=labels,
            joints=joints,
            gripper=gripper,
        )
        saved = self._store.save(record, self._snapshot())
        logger.info(
            "teach: %s %s %s (%s)",
            labels.animal,
            labels.stage.value,
            labels.result.value,
            "holding" if gripper is not None and gripper.holding else "not holding",
        )
        return saved

    def _read_gripper(self) -> GripperReading:
        assert self._gripper is not None
        servo = self._gripper.status().servo
        target = self._gripper.target
        holding = self._gripper.calibration.is_holding(servo.position, target, self._holding_margin)
        return GripperReading(servo=servo, target=target, holding=holding)
