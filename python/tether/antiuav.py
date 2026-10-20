"""tether-antiuav: stress-test ID keeping on Anti-UAV300 drone clips, without a detector.

The labeled drone boxes of each visible-light clip are turned into a detection
file with failures added on purpose, the tracker command is run on it, and its
tracks are scored against the labels:

- Random deletion: each labeled box is deleted with the given probability
  (seeded, so runs repeat). Scored as "ID kept": of the frames whose box was
  given to the tracker, the share where the clip's most common ID holds the drone.
- Gaps: stretches of the given length in seconds (converted with each clip's
  frame rate) are blanked, each after SETTLE_SECONDS of detections. A gap counts
  as recovered if the ID that held the drone on the frame before it holds it
  again on the first frame the drone is labeled after it. Prediction error is
  how far the held track's prediction was from the drone on each gap frame.

A track holds the drone on a frame only if it is "tracked" and its box overlaps
the labeled box with IoU of at least MIN_IOU. Without the IoU rule, a track that
never dies would score 100% on single-drone clips. Results are saved as JSON.

The tracker is any command called as `<command> DETECTIONS TRACKS`, as for
tether-mot17. Its own settings (like how long a track may search) stay its own.
"""

import argparse
import json
import math
import subprocess
import sys
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tether.formats import Box, Detection, Track, load_tracks, save_detections
from tether.matching import iou

DEFAULT_WORK_DIR = Path("results/antiuav")
DRONE = 0  # class ID given to the labeled drone boxes
MIN_IOU = 0.5  # a track holds the drone only if it overlaps the labeled box at least this much
SETTLE_SECONDS = 2.0  # detections given before each gap, so the track is confirmed and moving
DROP_RATES = (0.0, 0.2)  # 0 is the baseline: IDs lost with nothing deleted
GAP_SECONDS = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Clip:
    name: str
    fps: float
    boxes: list[Box | None]  # the labeled drone per frame, None where it isn't labeled


def load_clip(clip_dir: str | Path) -> Clip:
    """Read `visible.json` ({exist, gt_rect as x, y, w, h}) and `visible.mp4`'s frame rate."""
    clip_dir = Path(clip_dir)
    labels = json.loads((clip_dir / "visible.json").read_text(encoding="utf-8"))
    # Trust `exist`: an absent drone's rect is [], [0, 0, 0, 0], or sometimes a leftover box.
    boxes = [_corners(rect) if exists else None for exists, rect in zip(labels["exist"], labels["gt_rect"])]
    video = cv2.VideoCapture(str(clip_dir / "visible.mp4"))
    fps, n_frames = video.get(cv2.CAP_PROP_FPS), int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if len(boxes) != n_frames:
        raise ValueError(f"{clip_dir}: {len(boxes)} labels, but visible.mp4 has {n_frames} frames")
    return Clip(name=clip_dir.name, fps=fps, boxes=boxes)


def to_detections(boxes: list[Box | None]) -> list[list[Detection]]:
    return [[Detection(box=b, score=1.0, cls=DRONE)] if b is not None else [] for b in boxes]


def drop_random(boxes: list[Box | None], rate: float, seed: int | Sequence[int]) -> list[Box | None]:
    """Delete each box with probability `rate`. The same seed always deletes the same boxes."""
    draws = np.random.default_rng(seed).random(len(boxes))
    return [None if d < rate else b for b, d in zip(boxes, draws)]


def drop_clip(clip: Clip, rate: float, seed: int) -> list[Box | None]:
    """`drop_random` seeded per clip by name, so a clip's deletions don't depend on which other clips run."""
    return drop_random(clip.boxes, rate, seed=[seed, zlib.crc32(clip.name.encode())])


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(1, round(seconds * fps))


def place_gaps(n_frames: int, gap_frames: int, settle_frames: int) -> list[tuple[int, int]]:
    """Gaps as [start, end) frames: settle, gap, settle, gap, ... Each leaves a frame after it."""
    gaps = []
    start = settle_frames
    while start + gap_frames < n_frames:
        gaps.append((start, start + gap_frames))
        start += gap_frames + settle_frames
    return gaps


def blank(boxes: list[Box | None], gaps: list[tuple[int, int]]) -> list[Box | None]:
    in_gap = {f for start, end in gaps for f in range(start, end)}
    return [None if f in in_gap else b for f, b in enumerate(boxes)]


def holders(drone: Box | None, tracks: list[Track]) -> set[int]:
    """IDs of the tracks holding the drone on this frame: "tracked", and IoU of at least MIN_IOU with it."""
    if drone is None:
        return set()
    return {t.id for t in tracks if t.status == "tracked" and iou(t.box, drone) >= MIN_IOU}


def id_kept(gt: list[Box | None], given: list[Box | None], frames: list[list[Track]]) -> dict:
    """Of the frames whose box was given to the tracker, how many the most common holding ID holds."""
    scored = [f for f, b in enumerate(given) if b is not None]
    counts = Counter(i for f in scored for i in holders(gt[f], frames[f]))
    kept = counts.most_common(1)[0][1] if counts else 0
    return {"kept": kept, "frames": len(scored)}


def id_kept_ceiling(given: list[Box | None], max_missing: int | None) -> dict:
    """The best `id_kept` any tracker could score on these boxes if it forgets a track after
    `max_missing` frames in a row without one (None: never forgets). The kept frames are
    the given ones in the longest stretch without a longer absence."""
    runs, run, absent = [], 0, 0
    for box in given:
        if box is None:
            absent += 1
            continue
        if max_missing is not None and absent > max_missing:
            runs.append(run)
            run = 0
        run, absent = run + 1, 0
    runs.append(run)
    return {"kept": max(runs), "frames": sum(runs)}


def gap_recovery(gt: list[Box | None], gaps: list[tuple[int, int]], frames: list[list[Track]]) -> dict:
    """Score each gap with the drone labeled just before it and at some frame after it.

    "held_before" counts the gaps where a track held the drone on the frame before;
    "recovered" those where the same ID holds it on the first labeled frame after.
    Prediction error is measured on held gaps only: on each gap frame where the drone
    is labeled, the held track's prediction from the frame before against the label.
    """
    result = {"gaps": 0, "held_before": 0, "recovered": 0, "center_errors": [], "ious": []}
    for k, (start, end) in enumerate(gaps):
        limit = gaps[k + 1][0] if k + 1 < len(gaps) else len(gt)
        returned = next((f for f in range(end, limit) if gt[f] is not None), None)
        if gt[start - 1] is None or returned is None:
            continue
        result["gaps"] += 1
        before = holders(gt[start - 1], frames[start - 1])
        if not before:
            continue
        result["held_before"] += 1
        result["recovered"] += bool(before & holders(gt[returned], frames[returned]))
        held = min(before)  # if duplicate tracks both held it, the oldest one's prediction
        for f in range(start, end):
            pred = next((t.predicted_box for t in frames[f - 1] if t.id == held), None)
            if gt[f] is not None and pred is not None:
                result["center_errors"].append(_center_distance(pred, gt[f]))
                result["ious"].append(iou(pred, gt[f]))
    return result


def evaluate(dataset_dir: str | Path, work_dir: str | Path, tracker_cmd: str, name: str,
             split: str = "test", drop_rates: Sequence[float] = DROP_RATES,
             gap_seconds: Sequence[float] = GAP_SECONDS, seed: int = 0) -> dict:
    """Run every condition on every clip of `split` and save to `work_dir/trackers/<name>/results.json`.

    Detection files go to `work_dir/detections/<condition>/`, so other trackers
    can be run on the very same input.
    """
    work_dir = Path(work_dir)
    out_dir = work_dir / "trackers" / name
    clips = [load_clip(d) for d in sorted(p for p in (Path(dataset_dir) / split).iterdir() if p.is_dir())]
    if not clips:
        raise FileNotFoundError(f"no clips in {Path(dataset_dir) / split}")

    def run(condition: str, clip: Clip, given: list[Box | None]) -> list[list[Track]]:
        dets = work_dir / "detections" / condition / f"{clip.name}.jsonl"
        tracks = out_dir / condition / f"{clip.name}.jsonl"
        for d in (dets.parent, tracks.parent):
            d.mkdir(parents=True, exist_ok=True)
        save_detections(dets, to_detections(given))
        subprocess.run(f'{tracker_cmd} "{dets}" "{tracks}"', shell=True, check=True)
        frames = load_tracks(tracks)
        if len(frames) != len(given):
            raise ValueError(f"{tracks}: {len(frames)} frames, but {dets} has {len(given)}")
        return frames

    deletion = []
    for rate in drop_rates:
        print(f"random deletion {rate:g}", flush=True)
        kept = frames = 0
        for clip in clips:
            given = drop_clip(clip, rate, seed)
            score = id_kept(clip.boxes, given, run(f"drop-{rate}-seed{seed}", clip, given))
            kept, frames = kept + score["kept"], frames + score["frames"]
        deletion.append({"drop_rate": rate, "kept": kept, "frames": frames, "id_kept": _percent(kept, frames)})

    gap_results = []
    for seconds in gap_seconds:
        print(f"gaps of {seconds:g} s", flush=True)
        scores = []
        for clip in clips:
            gaps = place_gaps(len(clip.boxes), seconds_to_frames(seconds, clip.fps),
                              seconds_to_frames(SETTLE_SECONDS, clip.fps))
            given = blank(clip.boxes, gaps)
            scores.append(gap_recovery(clip.boxes, gaps, run(f"gap-{seconds}s", clip, given)))
        counts = {key: sum(s[key] for s in scores) for key in ("gaps", "held_before", "recovered")}
        errors = [e for s in scores for e in s["center_errors"]]
        ious = [v for s in scores for v in s["ious"]]
        gap_results.append({"gap_seconds": seconds, **counts,
                            "recovery": _percent(counts["recovered"], counts["held_before"]),
                            "prediction_frames": len(errors),
                            "prediction_error_px": float(np.mean(errors)) if errors else None,
                            "prediction_iou": float(np.mean(ious)) if ious else None})

    results = {"tracker": name, "split": split, "clips": len(clips), "seed": seed,
               "random_deletion": deletion, "gaps": gap_results}
    results_file = out_dir / "results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {results_file}")
    return results


def _corners(rect: list[float]) -> Box:
    x, y, w, h = (float(v) for v in rect)
    return (x, y, x + w, y + h)


def _center_distance(a: Box, b: Box) -> float:
    """In pixels."""
    return math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def _percent(part: int, whole: int) -> float | None:
    """None when there is nothing to score, rather than a flattering 0/0."""
    return 100 * part / whole if whole else None


def _report(results: dict) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}%"

    lines = [f"{results['clips']} {results['split']} clips"]
    lines += [f"deleting {d['drop_rate']:.0%}: ID kept {pct(d['id_kept'])} of {d['frames']} frames"
              for d in results["random_deletion"]]
    for g in results["gaps"]:
        error = "n/a" if g["prediction_error_px"] is None else f"{g['prediction_error_px']:.1f} px"
        lines.append(f"gaps of {g['gap_seconds']:g} s: recovered {pct(g['recovery'])} of "
                     f"{g['held_before']} held ({g['gaps']} gaps), prediction error {error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-antiuav", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="clips to score (default test; keep it unseen if you tune on train or val)")
    parser.add_argument("--drop-rates", type=float, nargs="*", default=list(DROP_RATES),
                        help="share of boxes deleted at random, one run each (default 0 0.2; "
                             "0 is the baseline: IDs lost with nothing deleted)")
    parser.add_argument("--gap-seconds", type=float, nargs="*", default=list(GAP_SECONDS),
                        help="gap lengths in seconds, one run each (default 0.25 0.5 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random deletion (default 0)")
    parser.add_argument("--tracker", default=f'"{sys.executable}" -m tether.cli',
                        help="command run as `<command> DETECTIONS TRACKS` (default: the Python tracker)")
    parser.add_argument("--name", default="tether", help="tracker name for the results folder (default tether)")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help=f"where detections, tracks and results go (default {DEFAULT_WORK_DIR})")
    args = parser.parse_args(argv)

    results = evaluate(args.antiuav, args.work_dir, args.tracker, args.name, split=args.split,
                       drop_rates=args.drop_rates, gap_seconds=args.gap_seconds, seed=args.seed)
    print(_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
