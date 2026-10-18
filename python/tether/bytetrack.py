"""tether-bytetrack: run ByteTrack on a detection file and write a track file.

The baseline for the MOT17 comparison. It reads the same unfiltered detection
files as Tether and applies its own thresholds, so it keeps its second matching
pass over low-score detections. The ByteTrack code is Ultralytics' BYTETracker.
"""

import argparse
from types import SimpleNamespace

import numpy as np

from tether.formats import Detection, Track, load_detections, save_tracks

PERSON = 0  # COCO class ID; ByteTrack tracks one class, and MOT17 only scores people
# Ultralytics' defaults for YOLO detectors (ultralytics/cfg/trackers/bytetrack.yaml).
# The ByteTrack paper's MOT17 settings (0.6 high, 0.7 to start a track) were tuned for
# a YOLOX model trained on MOT17, whose scores run much higher than YOLO26's.
SETTINGS = {
    "track_high_thresh": 0.25,  # first matching pass uses detections scored at least this
    "track_low_thresh": 0.1,  # second pass uses those above this and below the high threshold
    "new_track_thresh": 0.25,  # an unmatched detection starts a track only if scored at least this
    "track_buffer": 30,  # frames a lost track is kept for re-finding (Tether's max_missing_frames is also 30)
    "match_thresh": 0.8,  # largest 1 - IoU (score-fused) a first-pass match may have
    "fuse_score": True,  # first-pass cost is 1 - IoU * score, as in ByteTrack's MOT17 runs
}


class _BoxesView:
    """The slice of Ultralytics' `Boxes` that BYTETracker reads: conf, cls, xywh, boolean indexing."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy, self.conf, self.cls = xyxy, conf, cls

    @property
    def xywh(self) -> np.ndarray:  # center x, center y, width, height
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_BoxesView":
        return _BoxesView(self.xyxy[mask], self.conf[mask], self.cls[mask])


def track_detections(frames: list[list[Detection]], low_score_pass: bool = True) -> list[list[Track]]:
    """Run ByteTrack over every frame, keeping only people.

    `low_score_pass=False` turns off the second matching pass (a diagnostic: it
    then uses exactly the detections Tether does).

    Each frame's output is the tracks ByteTrack reports, all "tracked", sorted by
    ID. Their boxes are ByteTrack's Kalman-filtered boxes. ByteTrack has no
    searching state in its output and keeps no path, so `predicted_box` is the
    box, `frames_since_seen` is 0 and `path` is empty.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker  # only needed here, and slow to import

    settings = SETTINGS if low_score_pass else {**SETTINGS, "track_low_thresh": SETTINGS["track_high_thresh"]}
    tracker = BYTETracker(SimpleNamespace(**settings))
    out = []
    for dets in frames:
        people = [d for d in dets if d.cls == PERSON]
        boxes = np.array([d.box for d in people], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d.score for d in people], dtype=np.float32)
        classes = np.full(len(people), PERSON, dtype=np.float32)
        results = tracker.update(_BoxesView(boxes, scores, classes))
        tracks = []
        for x1, y1, x2, y2, track_id, *_ in results:
            box = (float(x1), float(y1), float(x2), float(y2))
            tracks.append(Track(id=int(track_id), box=box, cls=PERSON, status="tracked", predicted_box=box,
                                frames_since_seen=0, path=()))
        out.append(sorted(tracks, key=lambda t: t.id))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-bytetrack", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detections", help="input detection file (.jsonl)")
    parser.add_argument("tracks", help="output track file (.jsonl)")
    parser.add_argument("--no-low-score-pass", action="store_true",
                        help="turn off the second matching pass over low-score detections (a diagnostic)")
    args = parser.parse_args(argv)

    save_tracks(args.tracks, track_detections(load_detections(args.detections),
                                              low_score_pass=not args.no_low_score_pass))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
