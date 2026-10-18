"""Score a saved MOT17 detection set on its own, with no tracker: recall and precision.

    python tools/detector_recall.py data/MOT17 results/mot17/detections/yolo26n_1280x736

Keeps people scored at least --min-score (Tether's own cutoff by default) and
scores them with the same TrackEval setup as the trackers, each detection as its
own one-frame track. So recall and precision are counted at IoU 0.5 under
MOT17's official rule: a box on a static person, distractor or reflection is
dropped rather than counted as a false alarm. This is the detector's ceiling:
no tracker that only reports detected boxes can find more people.
Results go to <work-dir>/detector_recall/<detection set>/results.json.
"""

import argparse
import json
from itertools import count
from pathlib import Path

from tether.formats import Detection, Track, load_detections
from tether.mot17 import DEFAULT_WORK_DIR, PERSON, save_mot_results, score
from tether.tracker import MIN_SCORE


def detections_as_tracks(frames: list[list[Detection]], min_score: float) -> list[list[Track]]:
    """Each person detection scored at least `min_score` becomes a "tracked" track with a new ID."""
    ids = count(1)
    return [[Track(id=next(ids), box=d.box, cls=PERSON, status="tracked", predicted_box=d.box,
                   frames_since_seen=0, path=())
             for d in dets if d.cls == PERSON and d.score >= min_score]
            for dets in frames]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mot17", help="the unzipped MOT17 folder (the one holding train/)")
    parser.add_argument("detections", help="folder of <sequence>.jsonl detection files")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help=f"lowest score counted (default {MIN_SCORE})")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help=f"where results go (default {DEFAULT_WORK_DIR})")
    args = parser.parse_args(argv)

    gt_dir, dets_dir = Path(args.mot17) / "train", Path(args.detections)
    name = dets_dir.name
    trackers_dir = Path(args.work_dir) / "detector_recall"
    (trackers_dir / name / "data").mkdir(parents=True, exist_ok=True)
    sequences = sorted(p.stem for p in dets_dir.glob("*.jsonl"))
    for seq in sequences:
        tracks = detections_as_tracks(load_detections(dets_dir / f"{seq}.jsonl"), args.min_score)
        save_mot_results(trackers_dir / name / "data" / f"{seq}.txt", tracks, cls=PERSON)

    results = {"detections": name, "min_score": args.min_score, **score(gt_dir, trackers_dir, name, sequences)}
    results_file = trackers_dir / name / "results.json"
    results_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{'':18}{'Recall':>8}{'Prec':>8}")
    for seq, s in [*results["sequences"].items(), ("combined", results["combined"])]:
        print(f"{seq:18}{s['Recall']:8.1f}{s['Precision']:8.1f}")
    print(f"saved to {results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
