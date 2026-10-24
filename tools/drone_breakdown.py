"""Where a detector finds the drone: recall by recording session, and the class it named.

    python tools/drone_breakdown.py data/Anti-UAV300 results/antiuav/detections/yolo26n_1280x736/val

For the class-agnostic COCO baseline (#42): per session of the split, the share
of drone frames found at the threshold (the spread shows how much a val of few
sessions can move), and for each found drone the class of the detection that
found it (its best-scored hit at IoU >= 0.5, as tether-drone-score counts it),
named from the model's metadata (#46), so class 0 is "drone" for the drone model.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tether.antiuav import MIN_IOU, Clip, load_clip
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split, session_of
from tether.class_names import COCO_NAMES, class_name, class_names_for, parse_names_flag
from tether.formats import Detection, load_detections
from tether.matching import iou
from tether.tracker import MIN_SCORE

def dump_class_names(detections: Path, flag: str | None) -> tuple[str, ...]:
    """The flag's names, else those of the model the folder's dump.json names, else COCO's (no dump.json)."""
    if flag is not None:
        return parse_names_flag(flag)
    dump = Path(detections) / "dump.json"
    if not dump.exists():
        return COCO_NAMES
    model = json.loads(dump.read_text(encoding="utf-8"))["model_path"]
    if not Path(model).exists():
        raise SystemExit(f"drone_breakdown: error: {model} (from {dump}) isn't there to read class names from; "
                         "pass --class-names")
    return class_names_for(model, None)


def breakdown(clips: list[tuple[Clip, list[list[Detection]]]], threshold: float,
              names: tuple[str, ...] = COCO_NAMES) -> dict:
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
                classes[class_name(best.cls, names)] += 1
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
    parser.add_argument("--class-names", metavar="NAMES",
                        help="comma-separated names for class IDs 0, 1, ... (default: the metadata of the model "
                             "dump.json names, else COCO's)")
    parser.add_argument("--out", help="results file (default <detections>/breakdown.json)")
    args = parser.parse_args(argv)

    detections = Path(args.detections)
    clips = [(load_clip(d), load_detections(detections / f"{d.name}.jsonl"))
             for d in clip_dirs(args.antiuav, load_split(args.split_file), args.split)]
    names = dump_class_names(detections, args.class_names)
    results = [breakdown(clips, t, names) for t in args.thresholds]
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
