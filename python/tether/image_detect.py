"""tether-image-detect: dump a detector's detections for a list of loose images (Seraphim, Drone-vs-Bird).

The images are hard-linked (copied where a link can't be made) into a numbered
sequence, <dump folder>/sequence/000000.jpg, 000001.jpg, ..., in list order, and
the detector is run once on it, as on a video:

    <detector> --model MODEL --det-min-score SCORE --dump DETECTIONS <sequence>/%06d.jpg

tether-pipeline (and tether-demo) read such a pattern through OpenCV like a
video file, so the same model, letterbox and preprocessing run on the images
as on Anti-UAV clips. Line i of DETECTIONS is image i of the list. The dump goes to
<work-dir>/detections/<model name>/<set>/detections.jsonl, beside a dump.json
naming the model, score floor, images and their sizes (width, height) for the
scorer (tether-drone-score-images).

A complete dump is skipped, a dump with the wrong number of lines is refused,
and so is a rerun into the same folder with other settings.
"""

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2

from tether.antiuav import DEFAULT_WORK_DIR
from tether.antiuav_detect import DEFAULT_DETECTOR, dumped_frames
from tether.demo import DEFAULT_MODEL
from tether.detector import DETECTOR_MIN_SCORE


def dump_images(images_dir: str | Path, names: Sequence[str], set_name: str, work_dir: str | Path,
                model: str | Path, detector: str, min_score: float) -> Path:
    """Dump detections for `names` (files in `images_dir`), in order; returns the dump folder."""
    images_dir = Path(images_dir)
    suffixes = {Path(n).suffix.lower() for n in names}
    if len(suffixes) != 1:
        raise ValueError(f"one file type per sequence, but the list has {sorted(suffixes)}")
    out = Path(work_dir) / "detections" / Path(model).stem / set_name
    out.mkdir(parents=True, exist_ok=True)
    info = {"model": Path(model).stem, "model_path": str(model), "min_score": min_score, "set": set_name,
            "detector": detector, "images_dir": str(images_dir), "images": list(names),
            "sizes": [_size(images_dir / n) for n in names]}
    info_file, dets = out / "dump.json", out / "detections.jsonl"
    if info_file.exists() and json.loads(info_file.read_text(encoding="utf-8")) != info:
        raise ValueError(f"{out} was dumped with other settings ({info_file}); use another --work-dir")
    info_file.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    if dumped_frames(dets) == len(names):
        return out
    sequence = out / "sequence"
    shutil.rmtree(sequence, ignore_errors=True)
    sequence.mkdir()
    suffix = suffixes.pop()
    for i, name in enumerate(names):
        _link(images_dir / name, sequence / f"{i:06d}{suffix}")
    subprocess.run(f'{detector} --model "{model}" --det-min-score {min_score} '
                   f'--dump "{dets}" "{sequence / f"%06d{suffix}"}"', shell=True, check=True)
    shutil.rmtree(sequence)
    if dumped_frames(dets) != len(names):
        raise ValueError(f"{dets}: {dumped_frames(dets)} images of detections, but the list has {len(names)}")
    return out


def _size(path: Path) -> list[int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"can't read image {path}")
    return [image.shape[1], image.shape[0]]


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # another volume, or a file system without hard links
        shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-image-detect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", help="folder holding the images")
    parser.add_argument("--list", required=True,
                        help="JSON file whose \"kept\" list names the images to run, in order "
                             "(from tether-transfer-sets, e.g. splits/seraphim-test.json)")
    parser.add_argument("--set", required=True, help="name for the dump folder, e.g. seraphim-test")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"YOLO26 ONNX model (default {DEFAULT_MODEL})")
    parser.add_argument("--det-min-score", type=float, default=DETECTOR_MIN_SCORE,
                        help=f"lowest detector score kept (default {DETECTOR_MIN_SCORE})")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR,
                        help="command run as `<command> --model M --det-min-score S --dump DETECTIONS PATTERN` "
                             "(default: tether-demo, on ONNX Runtime's CPU provider)")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help=f"default {DEFAULT_WORK_DIR}")
    args = parser.parse_args(argv)

    names = json.loads(Path(args.list).read_text(encoding="utf-8"))["kept"]
    out = dump_images(args.images, names, args.set, args.work_dir, args.model, args.detector, args.det_min_score)
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
