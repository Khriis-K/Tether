"""tether-empty-sky: list the Anti-UAV frames that count as empty sky, for the false-alarm row (ADR 0004).

"Drone labeled absent" isn't the same as empty sky. On the session-disjoint val,
most of the drone model's detections on drone-absent frames were the drone half
out of view, which the labels mark absent (docs/benchmarks/drone-model.md).
So a frame counts as empty sky only if:

- the drone is labeled absent on it, and
- no labeled drone is within MARGIN_SECONDS of it, before or after, in the
  same clip. The frames just outside the clip count as possibly holding the
  drone, since a clip is one piece of a longer recording.

The rule reads only labels and frame indices, so it can list test frames
without looking at them. It was fixed on val before any test run: the margin is
a round 2 s (40 frames at 20 fps), just past both the furthest drone-absent
detection #43 found from a labeled drone (36 frames, docs/benchmarks/
drone-model-absent-detections.json) and the tracker's 30-frame search.

The val and test lists go to one file (default splits/antiuav-empty-sky.json),
as half-open [start, end) frame ranges per clip, with the val frame counts
at a few other margins (val_sizing); tether-drone-score
--empty-sky counts false alarms on those frames only.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tether.antiuav import Clip, load_clip, seconds_to_frames
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.formats import Box

DEFAULT_OUT = Path("splits/antiuav-empty-sky.json")
MARGIN_SECONDS = 2.0
SIZING_SECONDS = (0.5, 1.0, 2.0, 5.0)  # margins whose val frame counts are saved, to show how the rule was sized
SPLITS = ("val", "test")
RULE = ("the drone is labeled absent, and no labeled drone is within margin_seconds before or after, in the "
        "same clip; the frames just outside the clip count as possibly holding the drone")


def empty_sky_frames(boxes: Sequence[Box | None], margin: int) -> list[int]:
    """Frames more than `margin` frames from any labeled drone, the frames at -1 and len(boxes) included."""
    drones = [-1, *(i for i, box in enumerate(boxes) if box is not None), len(boxes)]
    frames = []
    for before, after in zip(drones, drones[1:]):
        frames += range(before + margin + 1, after - margin)
    return frames


def to_ranges(frames: Sequence[int]) -> list[list[int]]:
    """Sorted frame indices as half-open [start, end) runs."""
    ranges = []
    for f in frames:
        if ranges and ranges[-1][1] == f:
            ranges[-1][1] = f + 1
        else:
            ranges.append([f, f + 1])
    return ranges


def empty_sky_split(clips: Sequence[Clip], seconds: float = MARGIN_SECONDS) -> dict:
    listed = {clip.name: empty_sky_frames(clip.boxes, seconds_to_frames(seconds, clip.fps)) for clip in clips}
    fps = {clip.name: clip.fps for clip in clips}
    return {"clips": {name: to_ranges(frames) for name, frames in listed.items()},
            "frames": sum(len(frames) for frames in listed.values()),
            "minutes": sum(len(frames) / fps[name] / 60 for name, frames in listed.items())}


def load_empty_sky(path: str | Path, split: str) -> dict[str, set[int]]:
    """Each clip's empty-sky frames in one split of a saved list."""
    clips = json.loads(Path(path).read_text(encoding="utf-8"))["splits"][split]["clips"]
    return {name: {f for start, end in ranges for f in range(start, end)} for name, ranges in clips.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-empty-sky", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"default {DEFAULT_OUT}")
    args = parser.parse_args(argv)

    split = load_split(args.split_file)
    clips = {name: [load_clip(d) for d in clip_dirs(args.antiuav, split, name)] for name in SPLITS}
    saved = {"rule": RULE, "margin_seconds": MARGIN_SECONDS, "split_file": Path(args.split_file).as_posix(),
             "seed": split["seed"],
             "val_sizing": {str(s): empty_sky_split(clips["val"], s)["frames"] for s in SIZING_SECONDS},
             "splits": {name: empty_sky_split(clips[name]) for name in SPLITS}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(saved, indent=1) + "\n")
    for name in SPLITS:
        print(f"{name}: {saved['splits'][name]['frames']} empty-sky frames, {saved['splits'][name]['minutes']:.2f} min")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
