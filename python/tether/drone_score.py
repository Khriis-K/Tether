"""tether-drone-score: score a detector's dumped detections on an Anti-UAV split (ADR 0004).

Every detection counts as "drone", whatever its class, so a COCO model is scored
class-agnostically. A detection finds the labeled drone at IoU >= MIN_IOU. On
each frame the drone is found by its best-scored overlapping detection; every
other detection is a false positive (a second box on the drone included), the
greedy-by-score rule COCO uses.

Reported:
- AP50: the area under the precision envelope over every score cut, with ties
  cut together. It only sees scores down to the dump's floor (tether-antiuav-detect
  --det-min-score), so it's comparable between dumps with the same floor.
- Recall, precision and F1 at two thresholds: the best-F1 one on this split
  (or a given one, e.g. frozen from val before test) and the tracker's 0.25.
  A threshold below the dump's floor is refused.
- Recall by box size at source resolution: tiny (area under 16^2 px), small
  (under 32^2), medium (under 96^2), large.
- False alarms per minute: detections on frames where the drone is labeled
  absent, over those frames' duration at each clip's own frame rate.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tether.antiuav import MIN_IOU, Clip, load_clip
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.formats import Detection, load_detections
from tether.matching import iou
from tether.tracker import MIN_SCORE

SIZES = (("tiny", 16 ** 2), ("small", 32 ** 2), ("medium", 96 ** 2), ("large", float("inf")))


@dataclass(frozen=True)
class Matches:
    """What scoring needs from one or more clips, at any threshold."""
    drones: list[tuple[float, float | None]]  # per labeled drone: box area, score of the detection that found it
    scores: list[float]  # every detection's score
    absent_scores: list[float]  # the scores of detections on drone-absent frames
    absent_minutes: float


def match_clip(clip: Clip, frames: list[list[Detection]], min_iou: float = MIN_IOU) -> Matches:
    if len(frames) != len(clip.boxes):
        raise ValueError(f"{clip.name}: {len(frames)} frames of detections, but {len(clip.boxes)} labeled")
    drones, scores, absent_scores = [], [], []
    for box, dets in zip(clip.boxes, frames):
        scores += [d.score for d in dets]
        if box is None:
            absent_scores += [d.score for d in dets]
            continue
        hits = [d.score for d in dets if iou(d.box, box) >= min_iou]
        drones.append(((box[2] - box[0]) * (box[3] - box[1]), max(hits) if hits else None))
    return Matches(drones, scores, absent_scores, (len(clip.boxes) - len(drones)) / clip.fps / 60)


def merge(matches: Sequence[Matches]) -> Matches:
    return Matches([d for m in matches for d in m.drones], [s for m in matches for s in m.scores],
                   [s for m in matches for s in m.absent_scores], sum(m.absent_minutes for m in matches))


def ap50(m: Matches) -> float | None:
    """None with no drone labeled; 0 with no detections."""
    if not m.drones:
        return None
    cuts, found, detections = _curve(m)
    if not len(cuts):
        return 0.0
    recall, precision = found / len(m.drones), found / detections
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at this recall or more
    return float(np.sum(np.diff(recall, prepend=0.0) * envelope))


def best_f1_threshold(m: Matches) -> float | None:
    """The score cut with the highest F1 (the higher cut on a tie); None with no detections."""
    cuts, found, detections = _curve(m)
    if not len(cuts):
        return None
    return float(cuts[np.argmax(2 * found / (detections + len(m.drones)))])  # argmax: first, so highest cut


def _curve(m: Matches) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct score, highest first, with the drones found and detections kept at each as the cut."""
    scores = np.sort(m.scores)
    found = np.sort([s for _, s in m.drones if s is not None])
    cuts = np.unique(scores)[::-1]
    return (cuts, len(found) - np.searchsorted(found, cuts, side="left"),
            len(scores) - np.searchsorted(scores, cuts, side="left"))


def at_threshold(m: Matches, threshold: float) -> dict:
    found = _found([s for _, s in m.drones], threshold)
    detections = sum(s >= threshold for s in m.scores)
    false_alarms = sum(s >= threshold for s in m.absent_scores)
    by_size = {}
    for (name, limit), low in zip(SIZES, (0, *(limit for _, limit in SIZES[:-1]))):
        in_size = [s for area, s in m.drones if low <= area < limit]
        hit = _found(in_size, threshold)
        by_size[name] = {"found": hit, "drones": len(in_size), "recall": _ratio(hit, len(in_size))}
    return {"threshold": threshold, "recall": _ratio(found, len(m.drones)), "precision": _ratio(found, detections),
            "f1": _ratio(2 * found, detections + len(m.drones)),  # 2TP / (2TP + FP + FN)
            "found": found, "drones": len(m.drones), "detections": detections, "recall_by_size": by_size,
            "false_alarms": false_alarms, "absent_minutes": m.absent_minutes,
            "false_alarms_per_minute": false_alarms / m.absent_minutes if m.absent_minutes else None}


def evaluate(root: str | Path, split_file: str | Path, name: str, detections: str | Path,
             threshold: float | None = None) -> dict:
    """Score `detections/<clip>.jsonl` for every clip of split `name`."""
    split, detections = load_split(split_file), Path(detections)
    dump_info = detections / "dump.json"
    dump = json.loads(dump_info.read_text(encoding="utf-8")) if dump_info.exists() else {}
    m = merge([match_clip(load_clip(d), load_detections(detections / f"{d.name}.jsonl"))
               for d in clip_dirs(root, split, name)])
    chosen_by = "given" if threshold is not None else "best F1 on this split"
    threshold = threshold if threshold is not None else best_f1_threshold(m)
    rows = [{**at_threshold(m, threshold), "chosen_by": chosen_by}] if threshold is not None else []
    rows.append({**at_threshold(m, MIN_SCORE), "chosen_by": "the tracker's min score"})
    floor = dump.get("min_score")
    if floor is not None and any(row["threshold"] < floor for row in rows):
        # Detections under the floor were never saved, so recall there would be understated.
        raise ValueError(f"thresholds {[row['threshold'] for row in rows]} go below the dump's floor {floor}")
    return {"model": dump.get("model", detections.name), "detections": str(detections),
            "split": {"file": str(split_file), "name": name, "seed": split["seed"]},
            "clips": len(split["clips"][name]), "min_iou": MIN_IOU,
            "match_rule": "class-agnostic; the drone is found by its best-scored detection at IoU >= min_iou, "
                          "every other detection is a false positive",
            "min_score": floor, "ap50": ap50(m), "thresholds": rows}


def _found(hit_scores: list[float | None], threshold: float) -> int:
    """Drones found by a detection scored at least `threshold`."""
    return sum(s is not None and s >= threshold for s in hit_scores)


def _ratio(part: int, whole: int) -> float | None:
    """None when there is nothing to score, rather than a flattering 0/0."""
    return part / whole if whole else None


def _report(results: dict) -> str:
    def num(value: float | None, fmt: str) -> str:
        return "n/a" if value is None else format(value, fmt)

    lines = [f"{results['model']} on {results['split']['name']} ({results['clips']} clips): "
             f"AP50 {num(results['ap50'], '.3f')}"]
    for row in results["thresholds"]:
        sizes = ", ".join(f"{k} {num(v['recall'], '.3f')} ({v['drones']})" for k, v in row["recall_by_size"].items())
        lines.append(f"at {row['threshold']:.3f} ({row['chosen_by']}): recall {num(row['recall'], '.3f')}, "
                     f"precision {num(row['precision'], '.3f')}, F1 {num(row['f1'], '.3f')}, "
                     f"false alarms {num(row['false_alarms_per_minute'], '.1f')}/min; recall by size: {sizes}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-drone-score", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("detections", help="folder of <clip>.jsonl detection files (from tether-antiuav-detect)")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="default val")
    parser.add_argument("--threshold", type=float,
                        help="score threshold to report (default: the best-F1 one on this split)")
    parser.add_argument("--out", help="results file (default <detections>/results.json)")
    args = parser.parse_args(argv)

    results = evaluate(args.antiuav, args.split_file, args.split, args.detections, args.threshold)
    out = Path(args.out or Path(args.detections) / "results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    print(_report(results))
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
