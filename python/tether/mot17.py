"""tether-mot17: score a tracker on the MOT17 train set with HOTA, IDF1 and ID switches.

For each MOT17 train video: dump the detector's unfiltered detections (once;
later runs reuse them), run the tracker command on them, convert its track file
to the MOT results format, and score everything with TrackEval. Per-sequence and
combined results are saved as JSON.

The tracker is any command called as `<command> DETECTIONS TRACKS` that reads a
detection file and writes a track file (see docs/FORMATS.md), so the Python
tracker, the C++ tracker and ByteTrack are all scored the same way.
"""

import argparse
import configparser
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from tether.demo import DEFAULT_MODEL, dump_detections
from tether.detector import OnnxDetector
from tether.formats import Detection, Track, load_detections, load_tracks

DEFAULT_WORK_DIR = Path("results/mot17")
PERSON = 0  # COCO class ID; MOT17 only scores pedestrians
# Every MOT17 video comes three times, once per public detector (DPM, FRCNN, SDP),
# with the same frames and ground truth. We bring our own detections, so one copy each.
SEQUENCE_GLOB = "*-FRCNN"


def save_mot_results(path: str | Path, frames: list[list[Track]], cls: int) -> None:
    """Write tracks in the MOT results format: `frame,id,left,top,width,height,conf,-1,-1,-1`.

    Frames count from 1. Only "tracked" tracks of class `cls` are written: a
    tentative track isn't confirmed yet, and a searching track's box is stale.
    Boxes are written as-is, without MOT's 1-based pixel shift, as ByteTrack does.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, tracks in enumerate(frames):
            for t in tracks:
                if t.status != "tracked" or t.cls != cls:
                    continue
                x1, y1, x2, y2 = t.box
                f.write(f"{i + 1},{t.id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n")


def score(gt_dir: Path, trackers_dir: Path, tracker: str, sequences: list[str]) -> dict:
    """Score `trackers_dir/tracker/data/<sequence>.txt` against `gt_dir/<sequence>/gt/gt.txt`.

    HOTA, IDF1, Recall and Precision are percentages; IDSW is a count. Recall and
    precision count a box as found at IoU 0.5, after MOT17's rule drops boxes on
    static people, distractors and reflections. "combined" is TrackEval's
    score over all sequences together, not an average of the per-sequence ones.
    """
    import trackeval  # only needed here, and slow to import

    evaluator = trackeval.Evaluator({
        "USE_PARALLEL": False, "BREAK_ON_ERROR": True, "LOG_ON_ERROR": None,
        "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
    })
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trackers_dir), "TRACKERS_TO_EVAL": [tracker],
        "SKIP_SPLIT_FOL": True, "SEQ_INFO": {seq: None for seq in sequences}, "PRINT_CONFIG": False,
    })
    metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}),
               trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    results, _ = evaluator.evaluate([dataset], metrics)
    per_seq = results["MotChallenge2DBox"][tracker]

    def summary(seq: str) -> dict:
        r = per_seq[seq]["pedestrian"]
        return {"HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])),  # averaged over IoU thresholds
                "IDF1": 100 * float(r["Identity"]["IDF1"]),
                "IDSW": int(r["CLEAR"]["IDSW"]),
                "Recall": 100 * float(r["CLEAR"]["CLR_Re"]),  # at IoU 0.5
                "Precision": 100 * float(r["CLEAR"]["CLR_Pr"])}

    return {"sequences": {seq: summary(seq) for seq in sequences}, "combined": summary("COMBINED_SEQ")}


def evaluate(mot17_dir: str | Path, work_dir: str | Path, tracker_cmd: str, name: str,
             detect: Callable[[np.ndarray], list[Detection]], detector_name: str = "yolo26n") -> dict:
    """Run the whole evaluation and save it to `work_dir/trackers/<name>/results.json`.

    Detections go to `work_dir/detections/<detector_name>/` and are reused if
    already there, so every tracker is scored on the same input. `detect` is
    only called for sequences with no saved detections yet.
    """
    gt_dir = Path(mot17_dir) / "train"
    work_dir = Path(work_dir)
    dets_dir = work_dir / "detections" / detector_name
    trackers_dir = work_dir / "trackers"
    out_dir = trackers_dir / name
    for d in (dets_dir, out_dir / "tracks", out_dir / "data"):
        d.mkdir(parents=True, exist_ok=True)

    sequences = sorted(p.name for p in gt_dir.glob(SEQUENCE_GLOB))
    if not sequences:
        raise FileNotFoundError(f"no {SEQUENCE_GLOB} sequences in {gt_dir}")
    for seq in sequences:
        print(f"{seq}: ", end="", flush=True)
        dets = dets_dir / f"{seq}.jsonl"
        if not dets.exists():
            print("detecting, ", end="", flush=True)
            _dump_sequence(gt_dir / seq, detect, dets)
        print("tracking", flush=True)
        tracks = out_dir / "tracks" / f"{seq}.jsonl"
        subprocess.run(f'{tracker_cmd} "{dets}" "{tracks}"', shell=True, check=True)
        # MOT frames are numbered by line, so a tracker that drops frames would shift every later one.
        track_frames, det_frames = load_tracks(tracks), load_detections(dets)
        if len(track_frames) != len(det_frames):
            raise ValueError(f"{tracks}: {len(track_frames)} frames, but {dets} has {len(det_frames)}")
        save_mot_results(out_dir / "data" / f"{seq}.txt", track_frames, cls=PERSON)

    results = {"tracker": name, "detections": detector_name, **score(gt_dir, trackers_dir, name, sequences)}
    results_file = out_dir / "results.json"
    results_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {results_file}")
    return results


def _dump_sequence(seq_dir: Path, detect: Callable[[np.ndarray], list[Detection]], path: Path) -> None:
    """Detect every frame of a sequence. Written to a temp file first, so a run
    cut short never leaves a partial file that later runs would reuse."""
    info = configparser.ConfigParser()
    info.read(seq_dir / "seqinfo.ini")
    images = sorted((seq_dir / info["Sequence"]["imDir"]).glob("*" + info["Sequence"]["imExt"]))
    expected = int(info["Sequence"]["seqLength"])
    if len(images) != expected:
        raise ValueError(f"{seq_dir}: found {len(images)} frames, seqinfo.ini says {expected}")
    partial = path.with_suffix(".partial")
    dump_detections((cv2.imread(str(image)) for image in images), detect, partial)
    partial.replace(path)


def _report(results: dict) -> str:
    rows = [*results["sequences"].items(), ("combined", results["combined"])]
    lines = [f"{'':18}{'HOTA':>8}{'IDF1':>8}{'IDSW':>8}"]
    lines += [f"{seq:18}{s['HOTA']:8.2f}{s['IDF1']:8.2f}{s['IDSW']:8d}" for seq, s in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-mot17", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mot17", help="the unzipped MOT17 folder (the one holding train/)")
    parser.add_argument("--tracker", default=f'"{sys.executable}" -m tether.cli',
                        help="command run as `<command> DETECTIONS TRACKS` (default: the Python tracker)")
    parser.add_argument("--name", default="tether", help="tracker name for the results folder (default tether)")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help=f"where detections, tracks and results go (default {DEFAULT_WORK_DIR})")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"YOLO26 ONNX model (default {DEFAULT_MODEL}); its detections are saved "
                             "under its file name, so each model gets its own")
    args = parser.parse_args(argv)

    detector = None

    def detect(frame: np.ndarray) -> list[Detection]:
        # Built on first use, so scoring another tracker on saved detections needs no model.
        nonlocal detector
        detector = detector or OnnxDetector(args.model)
        return detector(frame)

    results = evaluate(args.mot17, args.work_dir, args.tracker, args.name, detect,
                       detector_name=Path(args.model).stem)
    print(_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
