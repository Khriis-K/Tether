"""tether-runtime-compare: the C++ pipeline on ONNX Runtime against TensorRT, on the same video.

Two checks, both through `tether-pipeline` (headless, sequential):

1. Agreement. Each runtime dumps its detections for every frame (--dump), and
   every detection scoring at least MIN_SCORE in one runtime must have a
   partner in the other: the same class, IoU at least MIN_IOU, and scores at
   most MAX_SCORE_DIFF apart. MIN_SCORE is the tracker's own threshold, so the
   check covers what the tracker acts on; detections just under it can cross
   it in either runtime and aren't counted. The runtimes agree when at least
   MIN_AGREEMENT of those detections have a partner.
2. Timing. Tracking runs on each runtime for several rounds, the order
   alternating each round, each run's timing log summarized with tether-timing's
   summary. The detect step is the model run: for TensorRT it includes copying
   the prepared frame to the GPU and the rows back.

Speedups are ONNX Runtime's time over TensorRT's (above 1: TensorRT is faster).
The results and every dump and timing log are saved together under --out.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tether.demo import DEFAULT_MODEL
from tether.formats import STEPS, Detection, load_detections, load_timings
from tether.matching import iou
from tether.parity import DEFAULT_CPP_DIR, REPO, cpp_binary
from tether.pipeline_compare import _run
from tether.timing import summarize

RUNTIMES = ("onnxruntime", "tensorrt")

# The tolerance, set before the first comparison on a real model.
MIN_SCORE = 0.25  # the tracker's MIN_SCORE
MIN_IOU = 0.95
MAX_SCORE_DIFF = 0.02
MIN_AGREEMENT = 0.99
MAX_EXAMPLES = 20  # detections without a partner kept in the results, for a look


def run_pipeline(cpp_dir: Path, video: str, model: str, runtime: str, extra: list, max_frames: int | None) -> None:
    binary = cpp_binary(cpp_dir, "tether-pipeline")
    if binary is None:
        raise SystemExit(f"tether-runtime-compare: error: no tether-pipeline in {cpp_dir}; "
                         "build it with cpp/build.sh Release")
    command = [binary, video, "--model", model, "--runtime", runtime, *extra]
    if max_frames is not None:
        command += ["--max-frames", max_frames]
    _run(command, f"tether-pipeline --runtime {runtime}")


def agreement(onnxruntime: list[list[Detection]], tensorrt: list[list[Detection]], min_score: float = MIN_SCORE,
              min_iou: float = MIN_IOU, max_score_diff: float = MAX_SCORE_DIFF) -> dict:
    """How closely two runtimes' detections agree, frame by frame, checked both ways.

    Each detection at or above `min_score` looks among the other runtime's
    detections of the same class (at any score) that overlap it by `min_iou`;
    its partner is the one with the closest score, and must be within
    `max_score_diff`. Partners aren't exclusive: two near-identical boxes in one
    runtime can share one in the other. Without such a box, it has no partner.
    """
    if len(onnxruntime) != len(tensorrt):
        raise ValueError(f"onnxruntime has {len(onnxruntime)} frames but tensorrt has {len(tensorrt)}")
    checked = partnered = 0
    worst_iou, worst_score_diff, worst_box_diff = 1.0, 0.0, 0.0
    examples = []
    for frame, both in enumerate(zip(onnxruntime, tensorrt)):
        for side in (0, 1):
            here, there = both[side], both[1 - side]
            for d in here:
                if d.score < min_score:
                    continue
                checked += 1
                overlaps = [(iou(d.box, o.box), o) for o in there if o.cls == d.cls]
                best_iou = max((v for v, _ in overlaps), default=0.0)
                worst_iou = min(worst_iou, best_iou)
                close = [o for v, o in overlaps if v >= min_iou]
                partner = min(close, key=lambda o: abs(d.score - o.score), default=None)
                score_diff = abs(d.score - partner.score) if partner else None  # None: nothing overlaps enough
                if partner and score_diff <= max_score_diff:
                    partnered += 1
                    worst_score_diff = max(worst_score_diff, score_diff)
                    worst_box_diff = max(worst_box_diff, max(abs(a - b) for a, b in zip(d.box, partner.box)))
                elif len(examples) < MAX_EXAMPLES:
                    examples.append({"frame": frame, "runtime": RUNTIMES[side], "box": list(d.box), "score": d.score,
                                     "cls": d.cls, "best_iou": best_iou, "score_diff": score_diff})
    return {"frames": len(onnxruntime), "checked": checked, "partnered": partnered,
            "agreement": partnered / checked if checked else 1.0,
            "worst_iou": worst_iou, "worst_partnered_score_diff": worst_score_diff,
            "worst_partnered_box_diff_px": worst_box_diff, "without_partner": examples}


def speedup(onnxruntime: dict, tensorrt: dict) -> dict:
    """ONNX Runtime's time over TensorRT's for FPS, frame time and each step; None where TensorRT's is 0."""
    def ratio(ort_ms: float, trt_ms: float) -> float | None:
        return ort_ms / trt_ms if trt_ms > 0 else None

    return {
        "fps": tensorrt["fps"] / onnxruntime["fps"],
        "frame_time": ratio(onnxruntime["frame_time_ms"]["mean"], tensorrt["frame_time_ms"]["mean"]),
        "steps": {s: ratio(onnxruntime["steps_ms"][s]["mean"], tensorrt["steps_ms"][s]["mean"]) for s in STEPS},
    }


def _report(result: dict) -> str:
    a = result["agreement"]
    lines = [f"{a['frames']} frames of {result['video']}",
             f"agreement: {a['partnered']}/{a['checked']} detections scoring >= {result['tolerance']['min_score']} "
             f"have a partner ({a['agreement']:.4f}; needs {result['tolerance']['min_agreement']}): "
             f"{'PASS' if result['agrees'] else 'FAIL'}",
             f"  partners: worst score difference {a['worst_partnered_score_diff']:.4f}, "
             f"worst box corner difference {a['worst_partnered_box_diff_px']:.3f} px"]
    for i, r in enumerate(result["rounds"], 1):
        ort, trt, x = r["onnxruntime"], r["tensorrt"], r["speedup"]
        lines.append(f"round {i} ({' then '.join(r['order'])}): detect {ort['steps_ms']['detect']['mean']:.2f} -> "
                     f"{trt['steps_ms']['detect']['mean']:.2f} ms ({x['steps']['detect']:.2f}x), "
                     f"FPS {ort['fps']:.1f} -> {trt['fps']:.1f} ({x['fps']:.2f}x)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-runtime-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file both runtimes run on")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"YOLO26 ONNX model, with its TensorRT engine next to it as .engine (default {DEFAULT_MODEL})")
    parser.add_argument("--rounds", type=int, default=2, help="timing pairs to run (default 2)")
    parser.add_argument("--max-frames", type=int, help="stop each run after this many frames (default: all)")
    parser.add_argument("--cpp-dir", type=Path, default=DEFAULT_CPP_DIR,
                        help="C++ build folder holding tether-pipeline, built with TensorRT (default: cpp/build)")
    parser.add_argument("--out", type=Path,
                        help="results file to write (default: results/runtime-compare-<date>-<time>.json in the repo)")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    out = args.out or REPO / "results" / f"runtime-compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    dumps = {}
    for runtime in RUNTIMES:
        dumps[runtime] = out.with_name(f"{out.stem}.{runtime}-detections.jsonl")
        run_pipeline(args.cpp_dir, args.video, args.model, runtime, ["--dump", dumps[runtime]], args.max_frames)
    agreed = agreement(*(load_detections(dumps[r]) for r in RUNTIMES))

    rounds = []
    for i in range(1, args.rounds + 1):
        order = list(RUNTIMES) if i % 2 else list(reversed(RUNTIMES))
        summaries = {}
        for runtime in order:
            log = out.with_name(f"{out.stem}.round{i}-{runtime}-timing.jsonl")
            run_pipeline(args.cpp_dir, args.video, args.model, runtime, ["--timing", log], args.max_frames)
            summaries[runtime] = summarize(load_timings(log))
        ort, trt = summaries["onnxruntime"], summaries["tensorrt"]
        if ort["frames"] != trt["frames"]:
            raise SystemExit(f"tether-runtime-compare: error: onnxruntime ran {ort['frames']} frames but tensorrt ran "
                             f"{trt['frames']}; they should read the same video")
        rounds.append({"order": order, "onnxruntime": ort, "tensorrt": trt, "speedup": speedup(ort, trt)})

    tolerance = {"min_score": MIN_SCORE, "min_iou": MIN_IOU, "max_score_diff": MAX_SCORE_DIFF,
                 "min_agreement": MIN_AGREEMENT}
    result = {"video": args.video, "model": args.model, "tolerance": tolerance, "agreement": agreed,
              "agrees": agreed["agreement"] >= MIN_AGREEMENT, "rounds": rounds}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(_report(result))
    print(f"saved {out}")
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
