"""tether-antiuav-detect: dump a detector's detections for every clip of an Anti-UAV split.

For each clip of the split (from the split file, see tether.antiuav_split), runs

    <detector> --model MODEL --det-min-score SCORE --dump DETECTIONS VIDEO

which both tether-demo (ONNX Runtime, the default) and tether-pipeline (the C++
pipeline, e.g. with `--runtime tensorrt` in the command) accept. The files go to
<work-dir>/detections/<model name>/<split>/<clip>.jsonl in the usual detection
file format, so the tracker can read the same files later, with a dump.json
beside them naming the model and score floor for the scorer (tether-drone-score).

Clips already dumped in full are skipped, so a run that was stopped picks up
where it left off. A file with the wrong number of frames is refused, and so is
a rerun into the same folder with other settings (its old files would be kept).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tether.antiuav import DEFAULT_WORK_DIR
from tether.antiuav_split import DEFAULT_SPLIT_FILE, clip_dirs, load_split
from tether.demo import DEFAULT_MODEL
from tether.detector import DETECTOR_MIN_SCORE
from tether.formats import load_detections

DEFAULT_DETECTOR = f'"{sys.executable}" -m tether.demo'


def dump_split(root: str | Path, split: dict, name: str, work_dir: str | Path, model: str | Path,
               detector: str, min_score: float) -> Path:
    """Dump every clip of split `name`; returns the folder of detection files."""
    out = Path(work_dir) / "detections" / Path(model).stem / name
    out.mkdir(parents=True, exist_ok=True)
    info = {"model": Path(model).stem, "model_path": str(model), "min_score": min_score, "split": name,
            "detector": detector}
    info_file = out / "dump.json"
    if info_file.exists() and json.loads(info_file.read_text(encoding="utf-8")) != info:
        # Its files would be kept and labeled with these settings.
        raise ValueError(f"{out} was dumped with other settings ({info_file}); use another --work-dir")
    info_file.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    clips = list(zip(split["clips"][name], clip_dirs(root, split, name)))
    for k, (clip, clip_dir) in enumerate(clips, 1):
        dets, expected = out / f"{clip_dir.name}.jsonl", split["frames"][clip]
        if _frames(dets) == expected:
            continue
        print(f"[{k}/{len(clips)}] {clip}", flush=True)
        subprocess.run(f'{detector} --model "{model}" --det-min-score {min_score} '
                       f'--dump "{dets}" "{clip_dir / "visible.mp4"}"', shell=True, check=True)
        if _frames(dets) != expected:
            raise ValueError(f"{dets}: {_frames(dets)} frames, but {clip} has {expected}")
    return out


def _frames(path: Path) -> int | None:
    """None for a missing file, or one a stopped run cut off mid-line."""
    try:
        return len(load_detections(path))
    except (FileNotFoundError, ValueError):  # json.JSONDecodeError is a ValueError
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-antiuav-detect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("antiuav", help="the unzipped Anti-UAV300 folder (the one holding train/, val/, test/)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"YOLO26 ONNX model (default {DEFAULT_MODEL})")
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE), help=f"default {DEFAULT_SPLIT_FILE}")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="clips to dump (default val; test is run once, at the end)")
    parser.add_argument("--det-min-score", type=float, default=DETECTOR_MIN_SCORE,
                        help=f"lowest detector score kept (default {DETECTOR_MIN_SCORE})")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR,
                        help="command run as `<command> --model M --det-min-score S --dump DETECTIONS VIDEO` "
                             "(default: tether-demo, on ONNX Runtime's CPU provider)")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help=f"default {DEFAULT_WORK_DIR}")
    args = parser.parse_args(argv)

    out = dump_split(args.antiuav, load_split(args.split_file), args.split, args.work_dir, args.model,
                     args.detector, args.det_min_score)
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
