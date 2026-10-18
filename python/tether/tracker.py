"""The tracker: one frame of detections in, the current tracks out.

Each track has a Kalman filter, and detections are matched against where each
track is predicted to be this frame, in this order:

1. Confirmed tracks (tracked and searching) against all detections, by IoU.
2. Tentative tracks against the detections left over, by IoU. Matching them
   second keeps a new track (say, from a duplicate box) from taking the
   detection of the confirmed track it overlaps.
3. A track seen only once has no velocity yet, so its prediction is just its
   last box; if IoU can't match it, it gets a second chance by center distance
   against whatever is still unmatched, so objects that are already fast when
   they appear keep their ID.

Detections scored below `min_score` are dropped before anything else, so
detection files can be saved unfiltered and each tracker picks its own cutoff.

Lifecycle, counted in frames: a new track is "tentative" until it has matched
`confirm_frames` frames in a row, then "tracked". A tentative track that misses
a frame is dropped, so one-frame false detections never get a confirmed ID. A
tracked track that misses is "searching": it keeps predicting, and can still be
matched, until it has missed more than `max_missing_frames` frames in a row.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tether.formats import Box, Detection, Point, Track
from tether.kalman import BoxFilter
from tether.matching import match, match_by_distance

PATH_LENGTH = 30  # box centers kept per track for trails
CONFIRM_FRAMES = 3  # matched frames in a row before a new track is confirmed
MAX_MISSING_FRAMES = 30  # missed frames in a row a confirmed track survives
MIN_SCORE = 0.25  # lowest detector confidence the tracker uses (Ultralytics' default; untuned)


@dataclass
class _TrackState:
    id: int
    box: Box
    cls: int
    motion: BoxFilter
    path: deque[Point] = field(default_factory=lambda: deque(maxlen=PATH_LENGTH))
    hits: int = 0  # detections matched so far, counting the first
    confirmed: bool = False
    frames_since_seen: int = 0

    def observe(self, box: Box, confirm_frames: int) -> None:
        self.box = box
        self.hits += 1
        self.frames_since_seen = 0
        self.confirmed = self.confirmed or self.hits >= confirm_frames
        self.path.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    @property
    def status(self) -> str:
        if not self.confirmed:
            return "tentative"
        return "tracked" if self.frames_since_seen == 0 else "searching"

    def snapshot(self) -> Track:
        return Track(
            id=self.id,
            box=self.box,
            cls=self.cls,
            status=self.status,
            predicted_box=self.motion.box,
            frames_since_seen=self.frames_since_seen,
            path=tuple(self.path),
        )


class Tracker:
    def __init__(self, confirm_frames: int = CONFIRM_FRAMES,
                 max_missing_frames: int = MAX_MISSING_FRAMES,
                 min_score: float = MIN_SCORE) -> None:
        self._confirm_frames = confirm_frames
        self._min_score = min_score
        self._max_missing_frames = max_missing_frames
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns the live tracks, sorted by ID."""
        detections = [d for d in detections if d.score >= self._min_score]
        # Every track's filter was stepped forward at the end of the last
        # update, so its box is already the prediction for this frame.
        allowed = np.array([[t.cls == d.cls for d in detections] for t in self._tracks], dtype=bool)
        allowed = allowed.reshape(len(self._tracks), len(detections))
        pairs = self._match_unmatched(lambda t: t.confirmed, match, detections, allowed, [])
        pairs += self._match_unmatched(lambda t: not t.confirmed, match, detections, allowed, pairs)
        pairs += self._match_unmatched(lambda t: t.hits == 1, match_by_distance,
                                       detections, allowed, pairs)

        survivors = []
        for ti, di in pairs:
            track = self._tracks[ti]
            track.motion.update(detections[di].box)
            track.observe(detections[di].box, self._confirm_frames)
            survivors.append(track)

        matched_tracks = {ti for ti, _ in pairs}
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.frames_since_seen += 1
                if track.confirmed and track.frames_since_seen <= self._max_missing_frames:
                    survivors.append(track)

        matched = {di for _, di in pairs}
        for di, det in enumerate(detections):
            if di not in matched:
                track = _TrackState(id=self._next_id, box=det.box, cls=det.cls,
                                    motion=BoxFilter(det.box))
                track.observe(det.box, self._confirm_frames)
                self._next_id += 1
                survivors.append(track)

        for track in survivors:
            track.motion.predict()
        self._tracks = sorted(survivors, key=lambda t: t.id)
        return [t.snapshot() for t in self._tracks]

    def _match_unmatched(self, wanted: Callable[[_TrackState], bool],
                         matcher: Callable[..., list[tuple[int, int]]],
                         detections: list[Detection], allowed: np.ndarray,
                         pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Pair the `wanted` tracks not yet in `pairs` with the detections not yet in `pairs`."""
        matched_tracks = {ti for ti, _ in pairs}
        matched_dets = {di for _, di in pairs}
        tis = [ti for ti, t in enumerate(self._tracks) if wanted(t) and ti not in matched_tracks]
        dis = [di for di in range(len(detections)) if di not in matched_dets]
        sub_pairs = matcher([self._tracks[ti].motion.box for ti in tis],
                            [detections[di].box for di in dis],
                            allowed[np.ix_(tis, dis)])
        return [(tis[a], dis[b]) for a, b in sub_pairs]
