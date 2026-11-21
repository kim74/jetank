"""
``jetank analyze``: run the camera model on saved images, e.g. recorded scenes.

Layer: app

Opens only the model (no camera, motors or servos), so it is safe while the robot stands
anywhere. Prints what was found in each image and how long the model took, and with
``--output`` writes each image with the floor tinted and the detections boxed.
"""

from __future__ import annotations

import contextlib
import logging
import statistics
import time
from typing import TYPE_CHECKING

from jetank.app.wiring import build_frame_analyzer
from jetank.domain.models import Frame
from jetank.errors import JetankError
from jetank.perception.overlay import draw_analysis

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jetank.config.settings import JetankSettings
    from jetank.domain.perception import FrameAnalysis

logger = logging.getLogger(__name__)


def cmd_analyze(settings: JetankSettings, images: Sequence[Path], output: Path | None) -> int:
    """Analyze each image and print the result; write annotated copies into ``output``.

    Raises:
        JetankError: No images, an image cannot be read or written, or the model cannot be
            loaded.
    """
    import cv2

    if not images:
        raise JetankError("no images given")
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    timings_ms: list[float] = []
    with contextlib.closing(build_frame_analyzer(settings.detector)) as analyzer:
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise JetankError(f"could not read the image {path}")
            start = time.perf_counter()
            analysis = analyzer.analyze(Frame(image=image, timestamp_s=time.monotonic()))
            timings_ms.append((time.perf_counter() - start) * 1000)
            print(f"{path.name}: {describe(analysis)}")
            if output is not None:
                target = output / path.name
                if not cv2.imwrite(str(target), draw_analysis(image, analysis)):
                    raise JetankError(f"could not write {target}")
    # The first run includes the GPU's warm-up, so it is left out of the typical time.
    typical = statistics.median(timings_ms[1:] or timings_ms)
    print(f"{len(images)} images, {typical:.0f} ms per image (median, preprocessing included)")
    if output is not None:
        print(f"annotated images in {output}")
    return 0


def describe(analysis: FrameAnalysis) -> str:
    """One line: the detections with confidences, and the share of floor."""
    found = ", ".join(f"{d.label} {d.confidence:.2f}" for d in analysis.detections) or "nothing"
    floor = "" if analysis.floor is None else f"; floor {analysis.floor.fraction:.0%} of the frame"
    return found + floor
