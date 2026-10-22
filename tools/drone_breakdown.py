"""Where a COCO detector finds the drone: recall by recording session, and the class it named.

    python tools/drone_breakdown.py data/Anti-UAV300 results/antiuav/detections/yolo26n_1280x736/val

For the class-agnostic COCO baseline (#42): per session of the split, the share
of drone frames found at the threshold (the spread shows how much a val of few
sessions can move), and for each found drone the COCO class of the detection
that found it (its best-scored hit at IoU >= 0.5, as tether-drone-score counts it).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tether.antiuav import MIN_IOU, Clip, load_clip
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split, session_of
from tether.formats import Detection, load_detections
from tether.matching import iou
from tether.tracker import MIN_SCORE

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush")


def breakdown(clips: list[tuple[Clip, list[list[Detection]]]], threshold: float) -> dict:
    sessions: dict[str, list[int]] = {}  # session -> [found, drone frames]
    classes = Counter()
    for clip, frames in clips:
        counts = sessions.setdefault(session_of(clip.name), [0, 0])
        for box, dets in zip(clip.boxes, frames):
            if box is None:
                continue
            counts[1] += 1
            hits = [d for d in dets if iou(d.box, box) >= MIN_IOU]
            best = max(hits, key=lambda d: d.score, default=None)
            if best is not None and best.score >= threshold:
                counts[0] += 1
                classes[COCO[best.cls]] += 1
    return {"threshold": threshold,
            "by_session": {s: {"found": f, "drones": n, "recall": f / n if n else None}
                           for s, (f, n) in sorted(sessions.items())},
            "by_class": dict(classes.most_common())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("detections", help="folder of <clip>.jsonl detection files (from tether-antiuav-detect)")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="default val")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.1, MIN_SCORE],
                        help=f"default 0.1 {MIN_SCORE}")
    parser.add_argument("--out", help="results file (default <detections>/breakdown.json)")
    args = parser.parse_args(argv)

    detections = Path(args.detections)
    clips = [(load_clip(d), load_detections(detections / f"{d.name}.jsonl"))
             for d in clip_dirs(args.antiuav, load_split(args.split_file), args.split)]
    results = [breakdown(clips, t) for t in args.thresholds]
    out = Path(args.out or detections / "breakdown.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(f"at {r['threshold']:g}: " + ", ".join(f"{s} {v['recall']:.3f} ({v['drones']})"
                                                    for s, v in r["by_session"].items()))
        print("  " + ", ".join(f"{k} {v}" for k, v in r["by_class"].items()))
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
