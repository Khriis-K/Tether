"""tether-transfer-sets: build the outside test sets for the drone model (ADR 0004).

  seraphim IMAGES       Seraphim test with Anti-UAV frames dropped (tether.antiuav_overlay):
                        writes the kept and dropped images (default splits/seraphim-test.json).
  birds IMAGES          Drone-vs-Bird bird test images (filenames starting "BT"), near
                        duplicates removed by pHash (tether.near_duplicates), in name order.
                        Ground truth is the filename: the label files call most birds
                        "Drone", so they're never read (default splits/drone-vs-bird-bt.json).
  overlay-check ANTIUAV Scores real Anti-UAV frames, sampled from the split's train and val
                        clips (never test), as Seraphim might store them, to show the overlay
                        check flags them all.

Building the sets runs no detector.
"""

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from tether.antiuav_overlay import OVERLAY_THRESHOLD, reticle_score, seraphim_like
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.near_duplicates import MAX_DISTANCE, dedup, natural_key, phash

BIRD_PREFIX = "BT"
HIGHEST_LISTED = 20  # kept Seraphim images listed with their score, for a look by eye


def _jpeg(image: np.ndarray, quality: int = 75) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_UNCHANGED)


# Forms an Anti-UAV frame (grayscale, 1920x1080) might take in a merged dataset.
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "as recorded": lambda f: f,
    "letterboxed 640, JPEG 75": lambda f: _jpeg(seraphim_like(f, 640)),
    "letterboxed 416, JPEG 75": lambda f: _jpeg(seraphim_like(f, 416)),
    "stretched to 640x640": lambda f: cv2.resize(f, (640, 640), interpolation=cv2.INTER_AREA),
    "mirrored, letterboxed 640": lambda f: seraphim_like(cv2.flip(f, 1), 640),
}


def scan_seraphim(images_dir: str | Path) -> dict:
    names = sorted((p.name for p in Path(images_dir).glob("*.jpg")), key=natural_key)
    scores = {name: reticle_score(cv2.imread(str(Path(images_dir) / name), cv2.IMREAD_GRAYSCALE)) for name in names}
    dropped = [{"image": n, "score": s} for n, s in scores.items() if s >= OVERLAY_THRESHOLD]
    kept = [n for n in names if scores[n] < OVERLAY_THRESHOLD]
    highest = sorted(kept, key=lambda n: -scores[n])[:HIGHEST_LISTED]
    return {"rule": "dropped if tether.antiuav_overlay.reticle_score >= threshold", "threshold": OVERLAY_THRESHOLD,
            "images": len(names), "kept_count": len(kept), "dropped_count": len(dropped), "dropped": dropped,
            "highest_kept": [{"image": n, "score": scores[n]} for n in highest], "kept": kept}


def dedup_birds(images_dir: str | Path) -> dict:
    names = sorted((p.name for p in Path(images_dir).glob(f"{BIRD_PREFIX}*.jpg")), key=natural_key)
    kept, duplicate_of = dedup([phash(cv2.imread(str(Path(images_dir) / n))) for n in names], MAX_DISTANCE)
    return {"rule": f"filenames starting {BIRD_PREFIX!r} (birds; label files ignored), in name order, each kept unless "
                    f"its pHash is within max_distance bits of an image already kept",
            "max_distance": MAX_DISTANCE, "images": len(names), "kept_count": len(kept),
            "kept": [names[i] for i in kept],
            "duplicates": {names[i]: names[k] for i, k in duplicate_of.items()}}


def overlay_check(frames: Sequence[np.ndarray]) -> dict:
    rows = {}
    for name, transform in TRANSFORMS.items():
        scores = [reticle_score(transform(f)) for f in frames]
        rows[name] = {"flagged": sum(s >= OVERLAY_THRESHOLD for s in scores), "min_score": min(scores),
                      "median_score": float(np.median(scores))}
    return {"threshold": OVERLAY_THRESHOLD, "frames": len(frames), "transforms": rows}


def _sample_frames(root: str | Path, split: dict, per_clip: int, seed: int) -> list[np.ndarray]:
    rng, frames = random.Random(seed), []
    for name in ("train", "val"):  # never test
        for clip in clip_dirs(root, split, name):
            video = cv2.VideoCapture(str(clip / "visible.mp4"))
            for index in sorted(rng.sample(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))), per_clip)):
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = video.read()
                if not ok:
                    raise ValueError(f"{clip}: can't read frame {index}")
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return frames


def _write(result: dict, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=1) + "\n")
    print(f"saved to {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-transfer-sets", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seraphim", help="drop Anti-UAV frames from Seraphim test")
    s.add_argument("images", help="Seraphim's test/images folder")
    s.add_argument("--out", default="splits/seraphim-test.json", help="default splits/seraphim-test.json")
    b = sub.add_parser("birds", help="deduplicate the Drone-vs-Bird bird test images")
    b.add_argument("images", help="Drone-vs-Bird's Dataset/test/images folder")
    b.add_argument("--out", default="splits/drone-vs-bird-bt.json", help="default splits/drone-vs-bird-bt.json")
    c = sub.add_parser("overlay-check", help="score real Anti-UAV train/val frames with the overlay check")
    c.add_argument("antiuav", help="the unzipped Anti-UAV300 folder")
    c.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    c.add_argument("--frames-per-clip", type=int, default=3, help="default 3")
    c.add_argument("--seed", type=int, default=0, help="default 0")
    c.add_argument("--out", default="docs/benchmarks/drone-transfer-overlay-check.json",
                   help="default docs/benchmarks/drone-transfer-overlay-check.json")
    args = parser.parse_args(argv)

    if args.command == "seraphim":
        result = scan_seraphim(args.images)
        print(f"{result['images']} images: {result['dropped_count']} dropped, {result['kept_count']} kept")
    elif args.command == "birds":
        result = dedup_birds(args.images)
        print(f"{result['images']} bird images: {result['kept_count']} kept after removing near duplicates")
    else:
        frames = _sample_frames(args.antiuav, load_split(args.split_file), args.frames_per_clip, args.seed)
        result = {"split_file": Path(args.split_file).as_posix(), "frames_per_clip": args.frames_per_clip, "seed": args.seed,
                  **overlay_check(frames)}
        for name, row in result["transforms"].items():
            print(f"{name}: {row['flagged']} of {result['frames']} flagged, lowest score {row['min_score']:.1f}")
    _write(result, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
