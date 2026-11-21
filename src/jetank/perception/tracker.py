"""
Following objects from frame to frame by box overlap.

Layer: perception

Each new detection continues the track of the same label it overlaps most (greedily, best
overlap first), or starts a new one. A track counts as *confirmed* after a few matches, which
filters out one-frame false detections, and is dropped once it has not been seen for a while.
Time comes from the frames, so the tracker needs no clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jetank.domain.detection import Track

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jetank.config.settings import TrackerConfig
    from jetank.domain.detection import Detection


class IouTracker:
    """Keeps tracks of the detections it is given, frame after frame.

    Args:
        config: Overlap needed to match, matches needed to confirm, age at which a track
            is dropped.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        """Tracks still alive, oldest first."""
        return list(self._tracks)

    def update(self, detections: Sequence[Detection], timestamp_s: float) -> list[Track]:
        """Match the detections of one frame to the tracks.

        Args:
            detections: What was found in the frame.
            timestamp_s: Capture time of the frame (monotonic seconds, not decreasing).

        Returns:
            Every track still alive, oldest first. Tracks seen in this frame have
            ``last_seen_s == timestamp_s``.

        Raises:
            ValueError: ``timestamp_s`` is older than a previous frame's.
        """
        if any(timestamp_s < track.last_seen_s for track in self._tracks):
            raise ValueError("frames must be given in capture order")
        pairs = sorted(
            (
                (track.box.iou(detection.box), t, d)
                for t, track in enumerate(self._tracks)
                for d, detection in enumerate(detections)
                if track.label == detection.label
            ),
            key=lambda pair: -pair[0],
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        tracks = list(self._tracks)
        for iou, t, d in pairs:
            if iou < self._config.match_iou_threshold:
                break
            if t in matched_tracks or d in matched_detections:
                continue
            matched_tracks.add(t)
            matched_detections.add(d)
            hits = tracks[t].hits + 1
            tracks[t] = tracks[t].model_copy(
                update={
                    "box": detections[d].box,
                    "confidence": detections[d].confidence,
                    "hits": hits,
                    "last_seen_s": timestamp_s,
                    "confirmed": hits >= self._config.confirm_hits,
                }
            )
        alive = [
            track for track in tracks if timestamp_s - track.last_seen_s <= self._config.max_age_s
        ]
        for d, detection in enumerate(detections):
            if d not in matched_detections:
                alive.append(self._start(detection, timestamp_s))
        self._tracks = alive
        return list(alive)

    def reset(self) -> None:
        """Forget every track (e.g. after the camera pose changed)."""
        self._tracks = []

    def _start(self, detection: Detection, timestamp_s: float) -> Track:
        track = Track(
            track_id=self._next_id,
            label=detection.label,
            box=detection.box,
            confidence=detection.confidence,
            hits=1,
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
            confirmed=self._config.confirm_hits <= 1,
        )
        self._next_id += 1
        return track
