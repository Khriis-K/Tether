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
  A threshold below the dump's floor is refused, and a best-F1 cut within
  AT_FLOOR_MARGIN of the lowest saved score is flagged (`at_lowest_saved_score`):
  F1 may still be rising below it, so the real best cut may be lower.
- Recall by box size at source resolution: tiny (area under 16^2 px), small
  (under 32^2), medium (under 96^2), large.
- False alarms per minute: detections on frames where the drone is labeled
  absent, over those frames' duration at each clip's own frame rate. With
  --empty-sky, only on the frames a tether-empty-sky list names.

tether-drone-score-images is the image-folder mode, for a tether-image-detect
dump: against YOLO-format labels (several drones per image, claimed greedily
by score) it reports the same AP50 and rows; on drone-free images (birds) it
reports the share of images with any detection at the threshold.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tether.antiuav import MIN_IOU, Clip, load_clip
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.empty_sky import load_empty_sky
from tether.formats import Box, Detection, load_detections
from tether.matching import iou
from tether.tracker import MIN_SCORE

AT_FLOOR_MARGIN = 0.01  # a best-F1 cut this close to the lowest saved score is flagged
SIZES = (("tiny", 16 ** 2), ("small", 32 ** 2), ("medium", 96 ** 2), ("large", float("inf")))


@dataclass(frozen=True)
class Matches:
    """What scoring needs from one or more clips, at any threshold."""
    drones: list[tuple[float, float | None]]  # per labeled drone: box area, score of the detection that found it
    scores: list[float]  # every detection's score
    absent_scores: list[float]  # the scores of detections on drone-absent frames
    absent_minutes: float


def match_clip(clip: Clip, frames: list[list[Detection]], min_iou: float = MIN_IOU,
               sky: set[int] | None = None) -> Matches:
    """With `sky` (tether.empty_sky's frames for this clip), false alarms count on those frames only."""
    if len(frames) != len(clip.boxes):
        raise ValueError(f"{clip.name}: {len(frames)} frames of detections, but {len(clip.boxes)} labeled")
    if sky is not None and any(clip.boxes[f] is not None for f in sky):
        raise ValueError(f"{clip.name}: an empty-sky frame has the drone labeled")
    drones, scores, absent_scores = [], [], []
    for i, (box, dets) in enumerate(zip(clip.boxes, frames)):
        scores += [d.score for d in dets]
        if box is None:
            if sky is None or i in sky:
                absent_scores += [d.score for d in dets]
            continue
        hits = [d.score for d in dets if iou(d.box, box) >= min_iou]
        drones.append((_area(box), max(hits) if hits else None))
    absent_frames = len(clip.boxes) - len(drones) if sky is None else len(sky)
    return Matches(drones, scores, absent_scores, absent_frames / clip.fps / 60)


def match_images(truth: list[list[Box]], frames: list[list[Detection]], min_iou: float = MIN_IOU) -> Matches:
    """Images with any number of drones each. Detections claim drones in score order, each taking the
    unclaimed drone it overlaps most at IoU >= min_iou (COCO's greedy rule); the rest are false positives.
    With one drone per image this is match_clip's rule."""
    if len(frames) != len(truth):
        raise ValueError(f"{len(frames)} images of detections, but {len(truth)} labeled")
    drones, scores = [], []
    for boxes, dets in zip(truth, frames):
        scores += [d.score for d in dets]
        found: list[float | None] = [None] * len(boxes)
        for d in sorted(dets, key=lambda d: -d.score):
            overlaps = [(iou(d.box, box), k) for k, box in enumerate(boxes) if found[k] is None]
            best = max(overlaps, default=(0.0, -1))
            if best[0] >= min_iou:
                found[best[1]] = d.score
        drones += [(_area(box), s) for box, s in zip(boxes, found)]
    return Matches(drones, scores, [], 0.0)


def drone_free_rate(frames: list[list[Detection]], threshold: float) -> dict:
    """On images with no drone: the share where any detection scores at least `threshold`."""
    with_alarm = sum(any(d.score >= threshold for d in dets) for dets in frames)
    return {"threshold": threshold, "images": len(frames), "with_alarm": with_alarm,
            "rate": _ratio(with_alarm, len(frames))}


def load_yolo_labels(path: str | Path, width: int, height: int) -> list[Box]:
    """A YOLO label file (class, center x, center y, width, height, all but the class as fractions of the
    image) as pixel corners. Classes are ignored."""
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cx, cy, w, h = (float(v) for v in line.split()[1:5])
            boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height))
    return boxes


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
             threshold: float | None = None, empty_sky: str | Path | None = None) -> dict:
    """Score `detections/<clip>.jsonl` for every clip of split `name`.

    With `empty_sky` (a tether-empty-sky list), false alarms count only on its frames for this split.
    """
    split, detections = load_split(split_file), Path(detections)
    dump_info = detections / "dump.json"
    dump = json.loads(dump_info.read_text(encoding="utf-8")) if dump_info.exists() else {}
    sky = load_empty_sky(empty_sky, name) if empty_sky is not None else None
    dirs = clip_dirs(root, split, name)
    if sky is not None and not {d.name for d in dirs} <= sky.keys():
        # Most likely a list for another split, which would score no false alarms in no time.
        raise ValueError(f"{empty_sky} lists no empty-sky frames for some {name} clips")
    m = merge([match_clip(load_clip(d), load_detections(detections / f"{d.name}.jsonl"),
                          sky=sky[d.name] if sky is not None else None) for d in dirs])
    floor = dump.get("min_score")
    return {"model": dump.get("model", detections.name), "detections": str(detections),
            "split": {"file": str(split_file), "name": name, "seed": split["seed"]},
            "clips": len(split["clips"][name]), "min_iou": MIN_IOU,
            "match_rule": "class-agnostic; the drone is found by its best-scored detection at IoU >= min_iou, "
                          "every other detection is a false positive",
            "empty_sky": str(empty_sky) if empty_sky is not None else None,
            "min_score": floor, "ap50": ap50(m), "thresholds": _rows(m, threshold, floor)}


def evaluate_images(detections: str | Path, labels: str | Path | None, threshold: float | None = None) -> dict:
    """Score a tether-image-detect dump: against YOLO-format `labels`, or, with labels None, as drone-free
    images (any detection is a false alarm)."""
    detections = Path(detections)
    dump = json.loads((detections / "dump.json").read_text(encoding="utf-8"))
    frames = load_detections(detections / "detections.jsonl")
    if len(frames) != len(dump["images"]):
        raise ValueError(f"{detections}: {len(frames)} images of detections, but {len(dump['images'])} listed")
    floor = dump["min_score"]
    results = {"model": dump["model"], "detections": str(detections), "set": dump["set"],
               "images": len(frames), "min_score": floor}
    if labels is None:
        rows = [drone_free_rate(frames, t) for t in (*([threshold] if threshold is not None else []), MIN_SCORE)]
        _check_floor(rows, floor)
        return {**results, "metric": "share of drone-free images with any detection at or above the threshold",
                "thresholds": rows}
    truth = [load_yolo_labels(Path(labels) / f"{Path(image).stem}.txt", w, h)
             for image, (w, h) in zip(dump["images"], dump["sizes"])]
    m = match_images(truth, frames)
    return {**results, "labels": str(labels), "min_iou": MIN_IOU,
            "match_rule": "class-agnostic; detections claim drones in score order, each the unclaimed drone it "
                          "overlaps most at IoU >= min_iou; every other detection is a false positive",
            "ap50": ap50(m), "thresholds": _rows(m, threshold, floor)}


def _rows(m: Matches, threshold: float | None, floor: float | None) -> list[dict]:
    """The given (or best-F1) threshold's row and the tracker's 0.25."""
    chosen_by = "given" if threshold is not None else "best F1 on this split"
    threshold = threshold if threshold is not None else best_f1_threshold(m)
    rows = [{**at_threshold(m, threshold), "chosen_by": chosen_by}] if threshold is not None else []
    rows.append({**at_threshold(m, MIN_SCORE), "chosen_by": "the tracker's min score"})
    for row in rows:  # a best-F1 cut here may not be the real best: F1 can still rise below what was saved
        row["at_lowest_saved_score"] = (row["chosen_by"] == "best F1 on this split"
                                        and row["threshold"] - min(m.scores) < AT_FLOOR_MARGIN)
    _check_floor(rows, floor)
    return rows


def _check_floor(rows: list[dict], floor: float | None) -> None:
    if floor is not None and any(row["threshold"] < floor for row in rows):
        # Detections under the floor were never saved, so recall there would be understated.
        raise ValueError(f"thresholds {[row['threshold'] for row in rows]} go below the dump's floor {floor}")


def _area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


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
    parser.add_argument("--empty-sky", help="count false alarms only on this tether-empty-sky list's frames "
                                            "(e.g. splits/antiuav-empty-sky.json)")
    parser.add_argument("--out", help="results file (default <detections>/results.json)")
    args = parser.parse_args(argv)

    results = evaluate(args.antiuav, args.split_file, args.split, args.detections, args.threshold, args.empty_sky)
    out = _save(results, args.out or Path(args.detections) / "results.json")
    print(_report(results))
    print(f"saved to {out}")
    return 0


def main_images(argv: list[str] | None = None) -> int:
    """tether-drone-score-images: the scorer's image-folder mode, on a tether-image-detect dump."""
    parser = argparse.ArgumentParser(
        prog="tether-drone-score-images",
        description="Score a detector on a folder of images (a tether-image-detect dump), class-agnostically: "
                    "against YOLO-format labels (AP50, recall, precision, F1, as tether-drone-score), or on "
                    "drone-free images as the share with any detection at the threshold.")
    parser.add_argument("detections", help="folder holding detections.jsonl and dump.json (from tether-image-detect)")
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--labels", help="folder of YOLO-format <image stem>.txt label files")
    truth.add_argument("--drone-free", action="store_true", help="the images hold no drone (e.g. bird images)")
    parser.add_argument("--threshold", type=float,
                        help="score threshold to report, e.g. frozen from val (default: the best-F1 one on these "
                             "images; drone-free images then get only the tracker's 0.25)")
    parser.add_argument("--out", help="results file (default <detections>/results.json)")
    args = parser.parse_args(argv)

    results = evaluate_images(args.detections, args.labels, args.threshold)
    out = _save(results, args.out or Path(args.detections) / "results.json")
    print(_report_images(results))
    print(f"saved to {out}")
    return 0


def _report_images(results: dict) -> str:
    lines = [f"{results['model']} on {results['set']} ({results['images']} images)"
             + (f": AP50 {results['ap50']:.3f}" if results.get("ap50") is not None else "")]
    for row in results["thresholds"]:
        if "rate" in row:
            lines.append(f"at {row['threshold']:.3f}: {row['with_alarm']} of {row['images']} images with a false alarm")
        else:
            lines.append(f"at {row['threshold']:.3f} ({row['chosen_by']}): {row['found']} of {row['drones']} drones "
                         f"found, {row['detections']} detections")
    return "\n".join(lines)


def _save(results: dict, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
