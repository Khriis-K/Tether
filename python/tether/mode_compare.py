"""tether-mode-compare: time the C++ pipeline sequential and parallel on the same video.

Runs `tether-pipeline --pipeline sequential` and `--pipeline parallel` (headless)
on the same video file with the same model, for several rounds. Which mode goes
first alternates each round, so a laptop speeding up or slowing down during the
runs doesn't land on one mode only. Each run's timing log is summarized with
tether-timing's summary; the summaries and the ratios are saved to --out, and
the timing logs are kept next to it.

Ratios: FPS and frame time are parallel's speedup (above 1: parallel is
faster). End-to-end delay and each step's time are parallel's over
sequential's (above 1: parallel takes longer).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tether.demo import DEFAULT_MODEL
from tether.formats import STEPS, load_timings
from tether.parity import DEFAULT_CPP_DIR, REPO, cpp_binary
from tether.pipeline_compare import _run
from tether.timing import summarize

MODES = ("sequential", "parallel")


def run_mode(cpp_dir: Path, video: str, model: str, mode: str, log: Path, max_frames: int | None) -> None:
    binary = cpp_binary(cpp_dir, "tether-pipeline")
    if binary is None:
        raise SystemExit(f"tether-mode-compare: error: no tether-pipeline in {cpp_dir}; "
                         "build it with cpp/build.sh Release")
    command = [binary, video, "--model", model, "--timing", log, "--pipeline", mode]
    if max_frames is not None:
        command += ["--max-frames", max_frames]
    _run(command, f"tether-pipeline --pipeline {mode}")


def compare_modes(sequential: dict, parallel: dict) -> dict:
    """Parallel's speedup in FPS and frame time; parallel over sequential for delay and each step.

    A step sequential spends no time on (headless draw) has no ratio: None.
    """
    def ratio(a: float, b: float) -> float | None:
        return a / b if b > 0 else None

    return {
        "fps": parallel["fps"] / sequential["fps"],
        "frame_time": ratio(sequential["frame_time_ms"]["mean"], parallel["frame_time_ms"]["mean"]),
        "end_to_end_delay": ratio(parallel["end_to_end_delay_ms"]["mean"], sequential["end_to_end_delay_ms"]["mean"]),
        "steps": {s: ratio(parallel["steps_ms"][s]["mean"], sequential["steps_ms"][s]["mean"]) for s in STEPS},
    }


def _report(result: dict) -> str:
    lines = [f"{result['rounds'][0]['sequential']['frames']} frames of {result['video']}"]
    for i, r in enumerate(result["rounds"], 1):
        seq, par, x = r["sequential"], r["parallel"], r["parallel_vs_sequential"]
        lines.append(f"round {i} ({' then '.join(r['order'])}): FPS {seq['fps']:.1f} -> {par['fps']:.1f} "
                     f"({x['fps']:.2f}x), delay {seq['end_to_end_delay_ms']['mean']:.1f} -> "
                     f"{par['end_to_end_delay_ms']['mean']:.1f} ms")
        steps = ", ".join(f"{s} {seq['steps_ms'][s]['mean']:.1f}->{par['steps_ms'][s]['mean']:.1f}" for s in STEPS)
        lines.append(f"  step means ms: {steps}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-mode-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file both modes run on")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"YOLO26 ONNX model (default {DEFAULT_MODEL})")
    parser.add_argument("--rounds", type=int, default=2, help="sequential-and-parallel pairs to run (default 2)")
    parser.add_argument("--max-frames", type=int, help="stop each run after this many frames (default: all)")
    parser.add_argument("--cpp-dir", type=Path, default=DEFAULT_CPP_DIR,
                        help="C++ build folder holding tether-pipeline (default: cpp/build)")
    parser.add_argument("--out", type=Path,
                        help="results file to write (default: results/mode-compare-<date>-<time>.json in the repo)")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    out = args.out or REPO / "results" / f"mode-compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rounds = []
    for i in range(1, args.rounds + 1):
        order = list(MODES) if i % 2 else list(reversed(MODES))
        summaries = {}
        for mode in order:
            log = out.with_name(f"{out.stem}.round{i}-{mode}-timing.jsonl")
            run_mode(args.cpp_dir, args.video, args.model, mode, log, args.max_frames)
            summaries[mode] = summarize(load_timings(log))
        seq, par = summaries["sequential"], summaries["parallel"]
        if seq["frames"] != par["frames"]:
            raise SystemExit(f"tether-mode-compare: error: sequential ran {seq['frames']} frames but parallel ran "
                             f"{par['frames']}; they should read the same video")
        rounds.append({"order": order, "sequential": seq, "parallel": par,
                       "parallel_vs_sequential": compare_modes(seq, par)})

    result = {"video": args.video, "model": args.model, "rounds": rounds}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(_report(result))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
