"""tether-runtime-compare: the C++ pipeline two ways on the same video.

`--compare runtimes` (the default) runs ONNX Runtime against TensorRT;
`--compare cuda-graph` runs TensorRT without its CUDA graph (--no-cuda-graph)
against TensorRT with it; `--compare frame-prep` runs TensorRT with CPU frame
prep against CUDA frame prep (--prep cuda). Two checks, both through
`tether-pipeline` (headless):

1. Agreement. Each side dumps its detections for every frame (--dump), and
   every detection scoring at least MIN_SCORE on one side must have a
   partner on the other: the same class, IoU at least MIN_IOU, and scores at
   most MAX_SCORE_DIFF apart. MIN_SCORE is the tracker's own threshold, so the
   check covers what the tracker acts on; detections just under it can cross
   it on either side and aren't counted. The sides agree when at least
   MIN_AGREEMENT of those detections have a partner. Whether the two dump
   files are byte-identical is saved too.
2. Timing. Tracking runs on each side for several rounds, in the --pipeline
   mode asked for, the order alternating each round, each run's timing log
   summarized with tether-timing's summary. The detect step is the model run:
   for TensorRT it includes copying the prepared frame to the GPU (or, from
   CUDA frame prep, from the prep's GPU buffer into the engine's) and the rows
   back.

Speedups are the first side's time over the second's (above 1: the second,
TensorRT, the CUDA graph or CUDA frame prep, is faster). The results and every dump and timing
log are saved together under --out.
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

# Each comparison's two sides: (name, --runtime, other tether-pipeline flags).
COMPARISONS = {
    "runtimes": (("onnxruntime", "onnxruntime", []), ("tensorrt", "tensorrt", [])),
    "cuda-graph": (("no-cuda-graph", "tensorrt", ["--no-cuda-graph"]), ("cuda-graph", "tensorrt", [])),
    "frame-prep": (("cpu-prep", "tensorrt", []), ("cuda-prep", "tensorrt", ["--prep", "cuda"])),
}
# The step each comparison changes, reported round by round.
CHANGED_STEP = {"runtimes": "detect", "cuda-graph": "detect", "frame-prep": "prep"}
RUNTIMES = tuple(name for name, _, _ in COMPARISONS["runtimes"])

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


def agreement(first: list[list[Detection]], second: list[list[Detection]], min_score: float = MIN_SCORE,
              min_iou: float = MIN_IOU, max_score_diff: float = MAX_SCORE_DIFF,
              names: tuple[str, str] = RUNTIMES) -> dict:
    """How closely two sides' detections agree, frame by frame, checked both ways.

    Each detection at or above `min_score` looks among the other side's
    detections of the same class (at any score) that overlap it by `min_iou`;
    its partner is the one with the closest score, and must be within
    `max_score_diff`. Partners aren't exclusive: two near-identical boxes on one
    side can share one on the other. Without such a box, it has no partner.
    `names` label the two sides in the result.
    """
    if len(first) != len(second):
        raise ValueError(f"{names[0]} has {len(first)} frames but {names[1]} has {len(second)}")
    checked = partnered = 0
    worst_iou, worst_score_diff, worst_box_diff = 1.0, 0.0, 0.0
    examples = []
    for frame, both in enumerate(zip(first, second)):
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
                    examples.append({"frame": frame, "runtime": names[side], "box": list(d.box), "score": d.score,
                                     "cls": d.cls, "best_iou": best_iou, "score_diff": score_diff})
    return {"frames": len(first), "checked": checked, "partnered": partnered,
            "agreement": partnered / checked if checked else 1.0,
            "worst_iou": worst_iou, "worst_partnered_score_diff": worst_score_diff,
            "worst_partnered_box_diff_px": worst_box_diff, "without_partner": examples}


def speedup(first: dict, second: dict) -> dict:
    """The first side's time over the second's (ONNX Runtime's over TensorRT's, or no graph's over the
    graph's) for FPS, frame time and each step; None where the second's is 0."""
    def ratio(first_ms: float, second_ms: float) -> float | None:
        return first_ms / second_ms if second_ms > 0 else None

    return {
        "fps": second["fps"] / first["fps"],
        "frame_time": ratio(first["frame_time_ms"]["mean"], second["frame_time_ms"]["mean"]),
        "steps": {s: ratio(first["steps_ms"][s]["mean"], second["steps_ms"][s]["mean"]) for s in STEPS},
    }


def _report(result: dict) -> str:
    a = result["agreement"]
    first, second = result["sides"]
    lines = [f"{a['frames']} frames of {result['video']}, {first} vs {second}, {result['pipeline']}",
             f"agreement: {a['partnered']}/{a['checked']} detections scoring >= {result['tolerance']['min_score']} "
             f"have a partner ({a['agreement']:.4f}; needs {result['tolerance']['min_agreement']}): "
             f"{'PASS' if result['agrees'] else 'FAIL'}",
             f"  partners: worst score difference {a['worst_partnered_score_diff']:.4f}, "
             f"worst box corner difference {a['worst_partnered_box_diff_px']:.3f} px; "
             f"dumps {'byte-identical' if result['dumps_identical'] else 'differ'}"]
    step = CHANGED_STEP[result["compare"]]
    for i, r in enumerate(result["rounds"], 1):
        a, b, x = r[first], r[second], r["speedup"]
        before, after = a["frame_time_ms"]["mean"], b["frame_time_ms"]["mean"]
        lines.append(f"round {i} ({' then '.join(r['order'])}): {step} {a['steps_ms'][step]['mean']:.2f} -> "
                     f"{b['steps_ms'][step]['mean']:.2f} ms ({x['steps'][step]:.2f}x), "
                     f"frame time {before:.2f} -> {after:.2f} ms ({before - after:.2f} ms saved), "
                     f"FPS {a['fps']:.1f} -> {b['fps']:.1f} ({x['fps']:.2f}x)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-runtime-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file both sides run on")
    parser.add_argument("--compare", choices=sorted(COMPARISONS), default="runtimes",
                        help="runtimes (default): ONNX Runtime vs TensorRT; cuda-graph: TensorRT without vs with "
                             "its CUDA graph; frame-prep: TensorRT with CPU vs CUDA frame prep")
    parser.add_argument("--pipeline", choices=["sequential", "parallel"], default="sequential",
                        help="pipeline mode of the timing runs (default sequential); the dumps are sequential")
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
    sides = COMPARISONS[args.compare]
    names = tuple(name for name, _, _ in sides)
    dumps = {}
    for name, runtime, flags in sides:
        dumps[name] = out.with_name(f"{out.stem}.{name}-detections.jsonl")
        run_pipeline(args.cpp_dir, args.video, args.model, runtime, [*flags, "--dump", dumps[name]], args.max_frames)
    agreed = agreement(*(load_detections(dumps[n]) for n in names), names=names)

    rounds = []
    for i in range(1, args.rounds + 1):
        order = list(sides) if i % 2 else list(reversed(sides))
        summaries = {}
        for name, runtime, flags in order:
            log = out.with_name(f"{out.stem}.round{i}-{name}-timing.jsonl")
            run_pipeline(args.cpp_dir, args.video, args.model, runtime,
                         [*flags, "--timing", log, "--pipeline", args.pipeline], args.max_frames)
            summaries[name] = summarize(load_timings(log))
        a, b = (summaries[n] for n in names)
        if a["frames"] != b["frames"]:
            raise SystemExit(f"tether-runtime-compare: error: {names[0]} ran {a['frames']} frames but {names[1]} ran "
                             f"{b['frames']}; they should read the same video")
        rounds.append({"order": [name for name, _, _ in order], names[0]: a, names[1]: b, "speedup": speedup(a, b)})

    tolerance = {"min_score": MIN_SCORE, "min_iou": MIN_IOU, "max_score_diff": MAX_SCORE_DIFF,
                 "min_agreement": MIN_AGREEMENT}
    identical = dumps[names[0]].read_bytes() == dumps[names[1]].read_bytes()
    result = {"video": args.video, "model": args.model, "compare": args.compare, "sides": list(names),
              "pipeline": args.pipeline, "tolerance": tolerance, "agreement": agreed,
              "agrees": agreed["agreement"] >= MIN_AGREEMENT, "dumps_identical": identical, "rounds": rounds}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(_report(result))
    print(f"saved {out}")
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
