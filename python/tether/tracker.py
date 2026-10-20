"""The tracker: one frame of detections in, the current tracks out.

Each track has a Kalman filter, and detections are matched against where each
track is predicted to be this frame, in this order:

1. Confirmed tracks (tracked and searching) against the high detections, by IoU.
2. Tracked tracks still unmatched against the low detections, by IoU of at
   least LOW_MIN_IOU. This keeps a track alive through a frame where its object
   is detected only weakly, say while partly occluded (ByteTrack's second pass).
3. Searching tracks still unmatched against the high detections left over, by
   center distance. A coasted box drifts off its object, so IoU stops seeing it;
   the gate is `search_gate + search_gate_growth * frames_since_seen` track box
   sizes, wider the longer the track has coasted. This runs before tentative
   tracks, or the tentative track the missed detection started would take it.
   The gate is capped at `search_gate_cap` times the distance to the nearest
   other confirmed track, so in a crowd it can't reach a neighbor's detection;
   without the cap every gate that helped Anti-UAV cost MOT17 HOTA and ID
   switches (docs/benchmarks/antiuav-search-gate-cap.md). Tentative tracks
   don't count, since the one a missed detection started is what the gate
   takes back. Center distance ignores size, so a detection more than
   SEARCH_GATE_MAX_SIZE_RATIO times taller or shorter than the track's last
   observed box is never a gate match: otherwise a track behind someone takes
   the box of the person walking in front.
4. Tentative tracks against the high detections left over, by IoU. Matching them
   after confirmed ones keeps a new track (say, from a duplicate box) from taking
   the detection of the confirmed track it overlaps.
5. A track seen only once has no velocity yet, so its prediction is just its
   last box; if IoU can't match it, it gets a second chance by center distance
   against the high detections still unmatched, so objects that are already fast when
   they appear keep their ID.

Merging duplicates: when the detection a track missed is too far for the gate,
it starts a tentative track, and once that is confirmed the object holds two
IDs. So when a tentative track is confirmed, an older confirmed track that
missed this frame takes it over if it is the same size (SEARCH_GATE_MAX_SIZE_RATIO,
against its last observed box) and its prediction is within its merge reach,
`merge_reach + merge_reach_growth * frames_since_seen` box sizes, capped the same
way as the gate. The older track keeps its ID and takes the new track's state;
the new ID is dropped. On Anti-UAV almost every lost ID was such a duplicate, and
the reach is far wider than the gate because the drone often reappears 5-10 box
sizes from where it was predicted; in a crowd the cap keeps it from reaching a
neighbor (docs/benchmarks/antiuav-duplicates.md).

Detections scored at least `min_score` are high; those above LOW_SCORE but below
`min_score` are low, and only ever used in stage 2, so they never start a track;
the rest are dropped. Detection files can be saved unfiltered and each tracker
picks its own cutoff.

Lifecycle, counted in frames: a new track is "tentative" until its net hits
(frames matched minus frames missed) reach `confirm_frames`, then "tracked". A
tentative track survives `tentative_misses` missed frames in a row and is dropped
on the next, so one deleted box early on doesn't start yet another ID, while a
one-frame false detection still never gets a confirmed ID: its net hits never
pass 1. With `tentative_misses` 0 it takes `confirm_frames` hits in a row. A
tracked track that misses is "searching": it keeps predicting, and can still be
matched, until it has missed more than `max_missing_frames` frames in a row.
Each missed frame its velocity is multiplied by `coast_damping` before the
prediction steps forward; 1 (the default) coasts at the last velocity, 0 holds
the last predicted position.
"""

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tether.formats import Box, Detection, Point, Track
from tether.kalman import BoxFilter
from tether.matching import MIN_IOU, center_distance, match, match_by_distance

PATH_LENGTH = 30  # box centers kept per track for trails
CONFIRM_FRAMES = 3  # net hits (matched minus missed frames) before a new track is confirmed
TENTATIVE_MISSES = 1  # missed frames in a row a tentative track survives
MAX_MISSING_FRAMES = 30  # missed frames in a row a confirmed track survives
MIN_SCORE = 0.25  # lowest detector confidence the tracker uses (Ultralytics' default; untuned)
# The low-score pass, with ByteTrack's published values (untuned):
LOW_SCORE = 0.1  # low detections score above this and below min_score
LOW_MIN_IOU = 0.5  # pairs with a low detection that overlap less are never matched
# Distance gate for searching tracks, in track box sizes; both 0 turns stage 3 off.
# Tuned on Anti-UAV val, checked on MOT17 (docs/benchmarks/antiuav-search-gate-cap.md).
SEARCH_GATE = 1.5  # after one missed frame the gate is SEARCH_GATE + SEARCH_GATE_GROWTH
SEARCH_GATE_GROWTH = 0.25  # added per missed frame
# The gate is at most this times the distance to the nearest other confirmed track; inf is no cap.
SEARCH_GATE_CAP = 0.5  # the issue's untuned starting point: two tracks' gates can't overlap
# A gate match's detection is at most this many times taller or shorter than the track's last box.
SEARCH_GATE_MAX_SIZE_RATIO = 1.5
# Merge reach for a duplicate, in box sizes; both 0 turns merging off. Tuned on Anti-UAV val, checked
# on MOT17 (docs/benchmarks/antiuav-duplicates.md).
MERGE_REACH = 10.0  # after one missed frame the reach is MERGE_REACH + MERGE_REACH_GROWTH
MERGE_REACH_GROWTH = 1.0  # added per missed frame
# Velocity is multiplied by this each frame a track goes unmatched; 1 is undamped, 0 holds still.
COAST_DAMPING = 1.0


@dataclass
class _TrackState:
    id: int
    box: Box
    cls: int
    motion: BoxFilter
    path: deque[Point] = field(default_factory=lambda: deque(maxlen=PATH_LENGTH))
    hits: int = 0  # detections matched so far, counting the first
    misses: int = 0  # frames missed while tentative
    confirmed: bool = False
    frames_since_seen: int = 0

    def observe(self, box: Box, confirm_frames: int) -> None:
        self.box = box
        self.hits += 1
        self.frames_since_seen = 0
        self.confirmed = self.confirmed or self.net_hits >= confirm_frames
        self.path.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    def take_over(self, other: "_TrackState") -> None:
        """Continue as `other`, which follows this track's object under a younger ID."""
        self.motion = other.motion
        self.hits += other.hits
        self.path.extend(other.path)

    @property
    def net_hits(self) -> int:
        return self.hits - self.misses

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
                 tentative_misses: int = TENTATIVE_MISSES,
                 max_missing_frames: int = MAX_MISSING_FRAMES,
                 min_score: float = MIN_SCORE, search_gate: float = SEARCH_GATE,
                 search_gate_growth: float = SEARCH_GATE_GROWTH,
                 search_gate_cap: float = SEARCH_GATE_CAP,
                 coast_damping: float = COAST_DAMPING, merge_reach: float = MERGE_REACH,
                 merge_reach_growth: float = MERGE_REACH_GROWTH) -> None:
        self._confirm_frames = confirm_frames
        self._tentative_misses = tentative_misses
        self._max_missing_frames = max_missing_frames
        self._min_score = min_score
        self._search_gate = search_gate
        self._search_gate_growth = search_gate_growth
        self._search_gate_cap = search_gate_cap
        self._coast_damping = coast_damping
        self._merge_reach = merge_reach
        self._merge_reach_growth = merge_reach_growth
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. Returns the live tracks, sorted by ID."""
        high = [d for d in detections if d.score >= self._min_score]
        low = [d for d in detections if LOW_SCORE < d.score < self._min_score]
        # High detections come first, so indices below len(high) are high ones.
        detections = high + low
        # Every track's filter was stepped forward at the end of the last
        # update, so its box is already the prediction for this frame.
        allowed = np.array([[t.cls == d.cls for d in detections] for t in self._tracks], dtype=bool)
        allowed = allowed.reshape(len(self._tracks), len(detections))
        is_low = np.arange(len(detections)) >= len(high)
        high_allowed, low_allowed = allowed & ~is_low, allowed & is_low
        pairs = self._match_unmatched(lambda t: t.confirmed, _by_iou(), detections, high_allowed, [])
        pairs += self._match_unmatched(lambda t: t.status == "tracked", _by_iou(LOW_MIN_IOU),
                                       detections, low_allowed, pairs)
        pairs += self._match_unmatched(lambda t: t.status == "searching" and self._gate(t) > 0,
                                       self._by_search_gate, detections, high_allowed, pairs)
        pairs += self._match_unmatched(lambda t: not t.confirmed, _by_iou(), detections,
                                       high_allowed, pairs)
        pairs += self._match_unmatched(lambda t: t.hits == 1, _by_one_hit_distance, detections,
                                       high_allowed, pairs)
        pairs, merged_away = self._merge_duplicates(pairs, detections)

        survivors = []
        for ti, di in pairs:
            track = self._tracks[ti]
            track.motion.update(detections[di].box)
            track.observe(detections[di].box, self._confirm_frames)
            survivors.append(track)

        matched_tracks = {ti for ti, _ in pairs} | merged_away
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.frames_since_seen += 1
                if not track.confirmed:
                    track.misses += 1
                limit = self._max_missing_frames if track.confirmed else self._tentative_misses
                if track.frames_since_seen <= limit:
                    survivors.append(track)

        matched = {di for _, di in pairs}
        for di, det in enumerate(high):
            if di not in matched:
                track = _TrackState(id=self._next_id, box=det.box, cls=det.cls,
                                    motion=BoxFilter(det.box))
                track.observe(det.box, self._confirm_frames)
                self._next_id += 1
                survivors.append(track)

        for track in survivors:
            track.motion.predict(self._coast_damping if track.frames_since_seen > 0 else 1.0)
        self._tracks = sorted(survivors, key=lambda t: t.id)
        return [t.snapshot() for t in self._tracks]

    def _gate(self, track: _TrackState) -> float:
        # frames_since_seen still counts up to the last frame: this one isn't decided yet.
        return self._search_gate + self._search_gate_growth * track.frames_since_seen

    def _reach(self, track: _TrackState) -> float:
        # Counted the same way as the gate's.
        return self._merge_reach + self._merge_reach_growth * track.frames_since_seen

    def _capped_gate(self, track: _TrackState) -> float:
        return self._capped(track, self._gate(track))

    def _capped(self, track: _TrackState, distance: float) -> float:
        """`distance`, at most search_gate_cap times the distance to the nearest other confirmed track."""
        if self._search_gate_cap == math.inf:  # no cap: skip the scan over every track
            return distance
        nearest = min((center_distance(track.motion.box, other.motion.box)
                       for other in self._tracks if other is not track and other.confirmed),
                      default=math.inf)
        return min(distance, self._search_gate_cap * nearest)

    def _merge_duplicates(self, pairs: list[tuple[int, int]],
                          detections: list[Detection]) -> tuple[list[tuple[int, int]], set[int]]:
        """Hand each pair that would confirm a tentative track to an older track it duplicates.

        Returns the new pairs and the tentative tracks merged away, which are
        dropped rather than counted as missing a frame.

        Runs before any pair is applied, so every box is still this frame's
        prediction and the tentative tracks don't count toward the cap. Only an
        older track takes over: a tentative track that survives misses can
        outlive a track born and confirmed after it, and the older ID is the
        one to keep. With confirm_frames 1 a track is born confirmed, so
        nothing is ever merged.
        """
        unmatched = set(range(len(self._tracks))) - {ti for ti, _ in pairs}
        merged, merged_away = [], set()
        for ti, di in pairs:
            new, box = self._tracks[ti], detections[di].box
            if not new.confirmed and new.net_hits + 1 >= self._confirm_frames:
                reachable = [(center_distance(self._tracks[oi].motion.box, box), oi) for oi in unmatched
                             if self._is_duplicate_of(self._tracks[oi], new, box)]
                if reachable:
                    _, oi = min(reachable)
                    unmatched.remove(oi)
                    self._tracks[oi].take_over(new)
                    merged_away.add(ti)
                    ti = oi
            merged.append((ti, di))
        return merged, merged_away

    def _is_duplicate_of(self, old: _TrackState, new: _TrackState, box: Box) -> bool:
        """Whether `new`, confirmed on `box`, is `old`'s object under a younger ID."""
        return (old.confirmed and old.id < new.id and old.cls == new.cls and _same_size(old.box, box)
                and center_distance(old.motion.box, box) <= self._capped(old, self._reach(old)))

    def _by_search_gate(self, tracks: list[_TrackState], boxes: list[Box],
                        allowed: np.ndarray) -> list[tuple[int, int]]:
        same_size = np.array([[_same_size(t.box, b) for b in boxes] for t in tracks], dtype=bool)
        return match_by_distance([t.motion.box for t in tracks], boxes,
                                 allowed & same_size.reshape(allowed.shape),
                                 max_distance=[self._capped_gate(t) for t in tracks])

    def _match_unmatched(self, wanted: Callable[[_TrackState], bool], matcher: _Matcher,
                         detections: list[Detection], allowed: np.ndarray,
                         pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Pair the `wanted` tracks not yet in `pairs` with the detections not yet in `pairs`."""
        matched_tracks = {ti for ti, _ in pairs}
        matched_dets = {di for _, di in pairs}
        tis = [ti for ti, t in enumerate(self._tracks) if wanted(t) and ti not in matched_tracks]
        dis = [di for di in range(len(detections)) if di not in matched_dets]
        sub_pairs = matcher([self._tracks[ti] for ti in tis],
                            [detections[di].box for di in dis],
                            allowed[np.ix_(tis, dis)])
        return [(tis[a], dis[b]) for a, b in sub_pairs]


# Pairs tracks with detection boxes, given which pairs are allowed; returns (track, detection) indices.
_Matcher = Callable[[list[_TrackState], list[Box], np.ndarray], list[tuple[int, int]]]


def _by_iou(min_iou: float = MIN_IOU) -> _Matcher:
    return lambda tracks, boxes, allowed: match([t.motion.box for t in tracks], boxes, allowed, min_iou)


def _same_size(last: Box, det: Box) -> bool:
    """Heights within SEARCH_GATE_MAX_SIZE_RATIO of each other; a zero height matches nothing."""
    a, b = last[3] - last[1], det[3] - det[1]
    return a > 0 and b > 0 and max(a / b, b / a) <= SEARCH_GATE_MAX_SIZE_RATIO


def _by_one_hit_distance(tracks: list[_TrackState], boxes: list[Box], allowed: np.ndarray) -> list[tuple[int, int]]:
    return match_by_distance([t.motion.box for t in tracks], boxes, allowed)
