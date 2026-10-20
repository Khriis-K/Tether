"""The best "ID kept" reachable on Anti-UAV, by how long a track searches: the numbers for #33.

    python tools/antiuav_ceiling.py data/Anti-UAV300 --split val

For each deletion rate, the boxes are deleted exactly as tether-antiuav deletes
them (same seed), and each search window is scored with `id_kept_ceiling`: the
best a tracker could do if it forgets a track after that many frames in a row
without a box. Windows in seconds are converted with each clip's frame rate.
No tracker is run. Results go to <work-dir>/ceiling/<split>.json.
"""

import argparse
import json
from pathlib import Path

from tether.antiuav import DEFAULT_WORK_DIR, DROP_RATES, drop_clip, id_kept_ceiling, load_clip, seconds_to_frames
from tether.tracker import MAX_MISSING_FRAMES

WINDOWS = (f"{MAX_MISSING_FRAMES} frames", "2 s", "3 s", "no limit")


def window_frames(window: str, fps: float) -> int | None:
    if window == "no limit":
        return None
    n, unit = window.split()
    return int(n) if unit == "frames" else seconds_to_frames(float(n), fps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--drop-rates", type=float, nargs="+", default=list(DROP_RATES))
    parser.add_argument("--seed", type=int, default=0, help="the seed tether-antiuav deleted with (default 0)")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help=f"where results go (default {DEFAULT_WORK_DIR})")
    args = parser.parse_args(argv)

    split_dir = Path(args.antiuav) / args.split
    clips = [load_clip(d) for d in sorted(p for p in split_dir.iterdir() if p.is_dir())]
    rows = []
    for rate in args.drop_rates:
        given = [drop_clip(clip, rate, args.seed) for clip in clips]
        for window in WINDOWS:
            scores = [id_kept_ceiling(g, window_frames(window, clip.fps)) for g, clip in zip(given, clips)]
            kept, frames = sum(s["kept"] for s in scores), sum(s["frames"] for s in scores)
            rows.append({"drop_rate": rate, "window": window, "kept": kept, "frames": frames,
                         "id_kept_ceiling": 100 * kept / frames})
            print(f"{rate:4.0%} deleted, {window:>9}: {100 * kept / frames:5.1f}% of {frames:,} frames", flush=True)

    results = {"split": args.split, "clips": len(clips), "seed": args.seed,
               "fps": sorted({clip.fps for clip in clips}), "ceilings": rows}
    out = Path(args.work_dir) / "ceiling" / f"{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
