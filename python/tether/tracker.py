"""The tracker: one frame of detections in, the current tracks out.

Tracer-bullet version: tracks match on their current box (no motion model yet),
every live track is "tracked", and a track that misses one frame is dropped.
The Kalman motion model and the full lifecycle replace those rules later.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tether.formats import Box, Detection, Point, Track
from tether.matching import match

PATH_LENGTH = 30  # box centers kept per track for trails


@dataclass
class _TrackState:
    id: int
    box: Box
    cls: int
    path: deque[Point] = field(default_factory=lambda: deque(maxlen=PATH_LENGTH))

    def observe(self, box: Box) -> None:
        self.box = box
        self.path.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    def snapshot(self) -> Track:
        return Track(
            id=self.id,
            box=self.box,
            cls=self.cls,
            status="tracked",
            predicted_box=self.box,  # no motion model yet
            frames_since_seen=0,
            path=tuple(self.path),
        )


class Tracker:
    def __init__(self) -> None:
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns the live tracks, sorted by ID."""
        allowed = np.array([[t.cls == d.cls for d in detections] for t in self._tracks], dtype=bool)
        pairs = match([t.box for t in self._tracks], [d.box for d in detections],
                      allowed.reshape(len(self._tracks), len(detections)))

        survivors = []
        for ti, di in pairs:
            self._tracks[ti].observe(detections[di].box)
            survivors.append(self._tracks[ti])

        matched = {di for _, di in pairs}
        for di, det in enumerate(detections):
            if di not in matched:
                track = _TrackState(id=self._next_id, box=det.box, cls=det.cls)
                track.observe(det.box)
                self._next_id += 1
                survivors.append(track)

        self._tracks = sorted(survivors, key=lambda t: t.id)
        return [t.snapshot() for t in self._tracks]
