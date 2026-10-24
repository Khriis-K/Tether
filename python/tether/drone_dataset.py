"""tether-drone-dataset: the drone detector's training set, every Nth frame of Anti-UAV as a YOLO dataset (ADR 0004).

Reads the split file (tether-antiuav-split) and writes, for its train and val
clips only (test is never read):

    <out>/images/{train,val}/<clip>_<frame>.jpg    every STRIDE-th frame from the first, resized to WIDTH
    <out>/labels/{train,val}/<clip>_<frame>.txt    one "0 cx cy w h" line (YOLO format), empty where the drone is absent
    <out>/data.yaml                                 one class, "drone", for Ultralytics
    <out>/dataset.json                              the settings and the image counts

Drone-absent frames are sampled at the same stride as the rest, so their share
stays near the source's (about 5%); they are the only in-domain negatives. The
val images are only for Ultralytics' own per-epoch check; the checkpoint is
chosen with tether-drone-score on every val frame.
"""

import argparse
import json
from pathlib import Path

import cv2

from tether.antiuav import load_clip
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.formats import Box

DEFAULT_STRIDE = 10  # half a second at 20 fps: neighbouring frames are near-copies
DEFAULT_WIDTH = 1280  # the model's input width; storing 1080p frames would only slow loading
SPLITS = ("train", "val")


def sampled_frames(n_frames: int, stride: int) -> range:
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    return range(0, n_frames, stride)


def yolo_label(box: Box | None, width: int, height: int) -> str:
    """A class-0 YOLO label line for a corner box clipped to the frame; "" for none."""
    if box is None:
        return ""
    x1, y1 = max(box[0], 0.0), max(box[1], 0.0)
    x2, y2 = min(box[2], float(width)), min(box[3], float(height))
    if x2 <= x1 or y2 <= y1:
        return ""
    return (f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def build_dataset(root: str | Path, split: dict, out: str | Path, stride: int = DEFAULT_STRIDE,
                  width: int = DEFAULT_WIDTH) -> dict:
    """Write the YOLO dataset; returns per split the images, absent images and the source's absent share."""
    out = Path(out)
    summary = {}
    for name in SPLITS:
        images, labels = out / "images" / name, out / "labels" / name
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        n_images = n_absent = source_frames = source_absent = 0
        for clip_dir in clip_dirs(root, split, name):
            clip = load_clip(clip_dir)
            source_frames += len(clip.boxes)
            source_absent += sum(b is None for b in clip.boxes)
            keep = set(sampled_frames(len(clip.boxes), stride))
            video = cv2.VideoCapture(str(clip_dir / "visible.mp4"))
            for i, box in enumerate(clip.boxes):
                if i not in keep:
                    video.grab()  # skips decoding the frame
                    continue
                ok, frame = video.read()
                if not ok:
                    raise ValueError(f"{clip_dir}: can't read frame {i}")
                h, w = frame.shape[:2]
                stem = f"{clip.name}_{i:06d}"
                small = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(images / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
                label = yolo_label(box, w, h)
                (labels / f"{stem}.txt").write_text(label + "\n" if label else "", encoding="utf-8")
                n_images += 1
                n_absent += not label
            video.release()
        summary[name] = {"images": n_images, "absent_images": n_absent,
                         "source_absent_share": source_absent / source_frames if source_frames else 0.0}
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "names:\n  0: drone\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-drone-dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("out", help="folder to write the dataset to (keep it out of git, e.g. under data/)")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"keep every Nth frame, drone or not (default {DEFAULT_STRIDE})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"resize images to this width, keeping the aspect ratio (default {DEFAULT_WIDTH})")
    args = parser.parse_args(argv)

    split = load_split(args.split_file)
    summary = build_dataset(args.antiuav, split, args.out, args.stride, args.width)
    info = {"split_file": args.split_file, "split_seed": split["seed"], "stride": args.stride,
            "width": args.width, **summary}
    Path(args.out, "dataset.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    for name, s in summary.items():
        print(f"{name}: {s['images']} images, {s['absent_images']} drone-absent "
              f"({s['absent_images'] / max(s['images'], 1):.1%}; source {s['source_absent_share']:.1%})")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
