"""Why IDs are lost on Anti-UAV: the numbers behind docs/benchmarks/antiuav-dropout.md.

    python tools/antiuav_diagnose.py data/Anti-UAV300 results/antiuav/detections/drop-0.2-seed0 \\
        results/antiuav/trackers/tether/drop-0.2-seed0

Given the detection and track files of one tether-antiuav condition, reports:

- Finding vs. keeping: the share of scored frames (box given to the tracker) where
  any track holds the drone, against the share the clip's most common ID holds,
  and the clip that used the most IDs.
- Reachable: the best ID-kept score possible when a track is forgotten after the
  tracker's MAX_MISSING_FRAMES frames without the drone labeled.
- Motion: how often a one-frame guess lands below the matcher's IoU 0.3 with the
  labeled drone, for a perfect constant-velocity guess and a stay-put guess
  (both from the labels alone), and for Tether's own prediction with nothing
  deleted (tracks that held the drone the frame before).

Results go to <work-dir>/diagnose/<condition>.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tether.antiuav import DEFAULT_WORK_DIR, Clip, holders, load_clip, to_detections
from tether.formats import Box, load_detections, load_tracks
from tether.matching import MIN_IOU as MATCH_IOU
from tether.matching import iou
from tether.tracker import MAX_MISSING_FRAMES, Tracker


def finding_vs_keeping(clips: list[Clip], dets_dir: Path, tracks_dir: Path) -> dict:
    scored = held_any = held_main = 0
    most_ids = (0, "", 0)  # IDs, clip, scored frames
    for clip in clips:
        given = load_detections(dets_dir / f"{clip.name}.jsonl")
        frames = load_tracks(tracks_dir / f"{clip.name}.jsonl")
        per_frame = [holders(clip.boxes[f], frames[f]) for f, dets in enumerate(given) if dets]
        counts = Counter(i for ids in per_frame for i in ids)
        scored += len(per_frame)
        held_any += sum(bool(ids) for ids in per_frame)
        held_main += counts.most_common(1)[0][1] if counts else 0
        most_ids = max(most_ids, (len(counts), clip.name, len(per_frame)))
    return {"scored_frames": scored, "held_by_any_track": 100 * held_any / scored,
            "held_by_most_common_id": 100 * held_main / scored,
            "most_ids": {"clip": most_ids[1], "ids": most_ids[0], "scored_frames": most_ids[2]}}


def reachable(clips: list[Clip], max_missing: int) -> dict:
    """The best "ID kept" any tracker that forgets a track after `max_missing` unlabeled
    frames could score with nothing deleted: per clip, the labeled frames in its longest
    stretch without a longer absence, over all labeled frames."""
    best = labeled = 0
    for clip in clips:
        runs, run, absent = [], 0, 0
        for box in clip.boxes:
            if box is None:
                absent += 1
                continue
            if absent > max_missing:
                runs.append(run)
                run = 0
            run, absent = run + 1, 0
        runs.append(run)
        best, labeled = best + max(runs), labeled + sum(runs)
    return {"max_missing_frames": max_missing, "labeled_frames": labeled, "id_kept_ceiling": 100 * best / labeled}


def guess_misses(clips: list[Clip]) -> dict:
    """Share of frames where a one-frame guess from the labels misses the drone (IoU below MATCH_IOU)."""
    frames = constant_velocity = stay_put = 0
    for clip in clips:
        b = clip.boxes
        for f in range(2, len(b)):
            if b[f - 2] is None or b[f - 1] is None or b[f] is None:
                continue
            guess: Box = tuple(2 * p - q for p, q in zip(b[f - 1], b[f - 2]))
            frames += 1
            constant_velocity += iou(guess, b[f]) < MATCH_IOU
            stay_put += iou(b[f - 1], b[f]) < MATCH_IOU
    return {"frames": frames, "constant_velocity": 100 * constant_velocity / frames,
            "stay_put": 100 * stay_put / frames}


def tether_misses(clips: list[Clip]) -> dict:
    """Same, for Tether's prediction, run on every labeled box (nothing deleted)."""
    frames = misses = 0
    for clip in clips:
        tracker, last = Tracker(), []
        for f, dets in enumerate(to_detections(clip.boxes)):
            tracks = tracker.update(dets)
            held = [t for t in last if t.id in holders(clip.boxes[f - 1], last)] if f else []
            if held and clip.boxes[f] is not None:
                frames += 1
                misses += max(iou(t.predicted_box, clip.boxes[f]) for t in held) < MATCH_IOU
            last = tracks
    return {"frames": frames, "tether": 100 * misses / frames}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder")
    parser.add_argument("detections", help="one condition's detection files, as written by tether-antiuav")
    parser.add_argument("tracks", help="the tracker's track files for that condition")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help=f"where results go (default {DEFAULT_WORK_DIR})")
    args = parser.parse_args(argv)

    split_dir = Path(args.antiuav) / args.split
    clips = [load_clip(d) for d in sorted(p for p in split_dir.iterdir() if p.is_dir())]
    results = {"split": args.split, "clips": len(clips), "frames": sum(len(c.boxes) for c in clips),
               "condition": Path(args.detections).name,
               "finding_vs_keeping": finding_vs_keeping(clips, Path(args.detections), Path(args.tracks)),
               "reachable": reachable(clips, MAX_MISSING_FRAMES),
               "one_frame_guess_misses": {**guess_misses(clips), "tether_no_deletion": tether_misses(clips)}}
    out = Path(args.work_dir) / "diagnose" / f"{results['condition']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
