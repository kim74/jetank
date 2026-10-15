"""Composition root and entry points."""

from jetank.app.wiring import Robot, build_camera, build_motor_driver, build_robot

__all__ = ["Robot", "build_camera", "build_motor_driver", "build_robot"]
