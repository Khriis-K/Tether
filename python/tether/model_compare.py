"""tether-model-compare: the C++ pipeline on TensorRT with several models, timed on the same video.

Runs `tether-pipeline --runtime tensorrt` (headless, CUDA graph on) once per
model per round, in the --pipeline mode asked for. The model order rotates each
round, so a laptop speeding up or slowing down during the runs doesn't land on
one model only. Each run's timing log is summarized with tether-timing's
summary, plus each step's busy share: its total time over the run's wall time.
In parallel mode the slowest stage, the one that sets the FPS, is near 100%.
Each model's exact input shape is read from the model, and each run is marked
against the spec's speed targets (TARGET_FPS, TARGET_P99_MS).

With --accuracy-dir, each model's MOT17 scores are added from
`<dir>/<prefix><model stem>/results.json` (written by tether-mot17), and the
combined table is printed as Markdown. The results and every timing log are
saved together under --out.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tether.formats import STEPS, StepTimes, load_timings
from tether.parity import DEFAULT_CPP_DIR, REPO, cpp_binary
from tether.pipeline_compare import _run
from tether.timing import summarize

# The spec's speed targets ("Targets" in docs/SPEC.md).
TARGET_FPS = 120.0
TARGET_P99_MS = 16.0


def model_order(models: list[str], round_number: int) -> list[str]:
    """Round 1 runs the models as given; each later round starts one model further on."""
    shift = (round_number - 1) % len(models)
    return models[shift:] + models[:shift]


def meets_targets(summary: dict) -> dict:
    """Whether one run meets each speed target: FPS at least TARGET_FPS, P99 frame time under TARGET_P99_MS."""
    fps = summary["fps"] >= TARGET_FPS
    p99 = summary["frame_time_ms"]["p99"] < TARGET_P99_MS
    return {"fps": fps, "p99": p99, "both": fps and p99}


def busy_share(frames: list[StepTimes]) -> dict:
    """Each step's total time over the run's wall time (first capture start to last draw end)."""
    wall = max(f["draw"][1] for f in frames) - min(f["capture"][0] for f in frames)
    return {s: sum(f[s][1] - f[s][0] for f in frames) / wall for s in STEPS}


def model_row(model: str, input_size: tuple[int, int], runs: list[dict], accuracy: dict | None) -> dict:
    """One table row: the model's shape, the range over its runs, and its MOT17 scores (or None).

    A target counts as met only if every run met it, so a row never claims
    more than its slowest run.
    """
    def span(values: list[float]) -> list[float]:
        return [min(values), max(values)]

    targets = [meets_targets(r) for r in runs]
    slowest = max(STEPS, key=lambda s: sum(r["busy_share"][s] for r in runs))
    h, w = input_size
    return {
        "model": model,
        "input": f"{w}x{h}",
        "runs": len(runs),
        "fps": span([r["fps"] for r in runs]),
        "frame_time_ms": span([r["frame_time_ms"]["mean"] for r in runs]),
        "frame_time_p99_ms": span([r["frame_time_ms"]["p99"] for r in runs]),
        "detect_ms": span([r["steps_ms"]["detect"]["mean"] for r in runs]),
        "slowest_stage": slowest,  # the highest busy share, summed over the runs
        "slowest_busy_share": span([r["busy_share"][slowest] for r in runs]),
        "meets_fps": all(t["fps"] for t in targets),
        "meets_p99": all(t["p99"] for t in targets),
        "meets_targets": all(t["both"] for t in targets),
        "mot17": None if accuracy is None else {k: accuracy["combined"][k] for k in ("HOTA", "IDF1", "IDSW")},
    }


def markdown_table(rows: list[dict]) -> str:
    """The combined table, one row per model (named by file stem): speed as min–max over the runs,
    MOT17 as one number (it's deterministic for a given detection file)."""
    def span(values: list[float]) -> str:
        lo, hi = values
        return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}–{hi:.2f}"

    def mark(ok: bool) -> str:
        return "yes" if ok else "no"

    def share(values: list[float]) -> str:
        lo, hi = (f"{100 * v:.1f}" for v in values)
        return f"{lo}%" if lo == hi else f"{lo}–{hi}%"

    lines = ["| Model | Input (w×h) | FPS | Frame time ms | P99 ms | Detect ms | Slowest stage (busy) "
             "| 120+ FPS | P99 < 16 ms | HOTA | IDF1 | ID switches |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        acc = r["mot17"]
        scores = f"{acc['HOTA']:.2f} | {acc['IDF1']:.2f} | {acc['IDSW']}" if acc else "— | — | —"
        lines.append(f"| {Path(r['model']).stem} | {r['input'].replace('x', '×')} | {span(r['fps'])} "
                     f"| {span(r['frame_time_ms'])} | {span(r['frame_time_p99_ms'])} | {span(r['detect_ms'])} "
                     f"| {r['slowest_stage']} ({share(r['slowest_busy_share'])}) "
                     f"| {mark(r['meets_fps'])} | {mark(r['meets_p99'])} | {scores} |")
    return "\n".join(lines)


def run_model(cpp_dir: Path, video: str, model: str, mode: str, log: Path, max_frames: int | None) -> None:
    binary = cpp_binary(cpp_dir, "tether-pipeline")
    if binary is None:
        raise SystemExit(f"tether-model-compare: error: no tether-pipeline in {cpp_dir}; "
                         "build it with TensorRT_ROOT set")
    command = [binary, video, "--model", model, "--runtime", "tensorrt", "--pipeline", mode, "--timing", log]
    if max_frames is not None:
        command += ["--max-frames", max_frames]
    _run(command, f"tether-pipeline --model {model}")


def _input_size(model: str) -> tuple[int, int]:
    from tether.detector import OnnxDetector  # loads ONNX Runtime, so only when a real run needs it

    return OnnxDetector(model).input_size


def _accuracy(accuracy_dir: Path | None, prefix: str, model: str) -> dict | None:
    if accuracy_dir is None:
        return None
    path = accuracy_dir / f"{prefix}{Path(model).stem}" / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-model-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file every model runs on")
    parser.add_argument("--models", nargs="+", required=True,
                        help="YOLO26 ONNX models, each with its TensorRT engine next to it (.engine)")
    parser.add_argument("--pipeline", choices=("sequential", "parallel"), default="parallel",
                        help="pipeline mode of the timing runs (default parallel)")
    parser.add_argument("--rounds", type=int, default=2, help="runs per model (default 2)")
    parser.add_argument("--max-frames", type=int, help="stop each run after this many frames (default: all)")
    parser.add_argument("--accuracy-dir", type=Path,
                        help="tether-mot17 trackers folder to read each model's MOT17 scores from")
    parser.add_argument("--accuracy-prefix", default="tether_",
                        help="tracker name before the model stem in --accuracy-dir (default tether_)")
    parser.add_argument("--cpp-dir", type=Path, default=DEFAULT_CPP_DIR,
                        help="C++ build folder holding tether-pipeline (default: cpp/build)")
    parser.add_argument("--out", type=Path,
                        help="results file to write (default: results/model-compare-<date>-<time>.json in the repo)")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    out = args.out or REPO / "results" / f"model-compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = {m: [] for m in args.models}
    orders = []
    for i in range(1, args.rounds + 1):
        order = model_order(args.models, i)
        orders.append(order)
        for model in order:
            log = out.with_name(f"{out.stem}.round{i}-{Path(model).stem}-timing.jsonl")
            run_model(args.cpp_dir, args.video, model, args.pipeline, log, args.max_frames)
            frames = load_timings(log)
            runs[model].append({**summarize(frames), "busy_share": busy_share(frames)})
            print(f"round {i}, {model}: {runs[model][-1]['fps']:.1f} FPS", flush=True)

    rows = [model_row(m, _input_size(m), runs[m], _accuracy(args.accuracy_dir, args.accuracy_prefix, m))
            for m in args.models]
    result = {"video": args.video, "pipeline": args.pipeline, "orders": orders,
              "targets": {"fps": TARGET_FPS, "p99_ms": TARGET_P99_MS}, "rows": rows, "runs": runs}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(markdown_table(rows))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
