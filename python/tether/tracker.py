"""The tracker: one frame of detections in, the current tracks out.

Each track has a Kalman filter, and detections are matched against where each
track is predicted to be this frame. A track seen only once has no velocity
yet, so its prediction is just its last box; if IoU can't match it, it gets a
second chance by center distance, so objects that are already fast when they
appear keep their ID. Still tracer-bullet lifecycle: every live
track is "tracked", and a track that misses one frame is dropped. The full
lifecycle replaces those rules later.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tether.formats import Box, Detection, Point, Track
from tether.kalman import BoxFilter
from tether.matching import match, match_by_distance

PATH_LENGTH = 30  # box centers kept per track for trails


@dataclass
class _TrackState:
    id: int
    box: Box
    cls: int
    motion: BoxFilter
    path: deque[Point] = field(default_factory=lambda: deque(maxlen=PATH_LENGTH))
    hits: int = 0  # detections matched so far, counting the first

    def observe(self, box: Box) -> None:
        self.box = box
        self.hits += 1
        self.path.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    def snapshot(self) -> Track:
        return Track(
            id=self.id,
            box=self.box,
            cls=self.cls,
            status="tracked",
            predicted_box=self.motion.box,
            frames_since_seen=0,
            path=tuple(self.path),
        )


class Tracker:
    def __init__(self) -> None:
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns the live tracks, sorted by ID."""
        # Every track's filter was stepped forward at the end of the last
        # update, so its box is already the prediction for this frame.
        allowed = np.array([[t.cls == d.cls for d in detections] for t in self._tracks], dtype=bool)
        allowed = allowed.reshape(len(self._tracks), len(detections))
        pairs = match([t.motion.box for t in self._tracks], [d.box for d in detections], allowed)
        pairs += self._match_new_tracks_by_distance(detections, allowed, pairs)

        survivors = []
        for ti, di in pairs:
            track = self._tracks[ti]
            track.motion.update(detections[di].box)
            track.observe(detections[di].box)
            survivors.append(track)

        matched = {di for _, di in pairs}
        for di, det in enumerate(detections):
            if di not in matched:
                track = _TrackState(id=self._next_id, box=det.box, cls=det.cls,
                                    motion=BoxFilter(det.box))
                track.observe(det.box)
                self._next_id += 1
                survivors.append(track)

        for track in survivors:
            track.motion.predict()
        self._tracks = sorted(survivors, key=lambda t: t.id)
        return [t.snapshot() for t in self._tracks]

    def _match_new_tracks_by_distance(self, detections: list[Detection], allowed: np.ndarray,
                                      pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Pair still-unmatched once-seen tracks with still-unmatched detections."""
        matched_tracks = {ti for ti, _ in pairs}
        matched_dets = {di for _, di in pairs}
        tis = [ti for ti, t in enumerate(self._tracks) if t.hits == 1 and ti not in matched_tracks]
        dis = [di for di in range(len(detections)) if di not in matched_dets]
        sub_pairs = match_by_distance([self._tracks[ti].motion.box for ti in tis],
                                      [detections[di].box for di in dis],
                                      allowed[np.ix_(tis, dis)])
        return [(tis[a], dis[b]) for a, b in sub_pairs]
