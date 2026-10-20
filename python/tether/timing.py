"""tether-timing: summarize a timing log into frame time, worst 1% and end-to-end delay.

Prints the summary and saves it as JSON next to the log (or to --out).
"""

import argparse
import json
from pathlib import Path

from tether.formats import STEPS, StepTimes, load_dropped, load_timings


def p99(values: list[float]) -> float:
    """Worst 1% value, by nearest rank: the smallest value that at least 99% of values are at or below.

    It's always one of the values. Under 100 values, that's the largest.
    """
    rank = (99 * len(values) + 99) // 100  # ceil(0.99 * n), in exact integer math
    return sorted(values)[rank - 1]


def summarize(frames: list[StepTimes]) -> dict:
    """Frame time, end-to-end delay and each step's time, as mean and P99 in milliseconds.

    Frame time is the gap between one frame finishing (draw end) and the next.
    End-to-end delay runs from a frame being in hand (capture end) to it being on
    screen (draw end). Both still hold when steps of different frames overlap.
    """
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames to measure a frame time, got {len(frames)}")
    finished = [f["draw"][1] for f in frames]
    frame_times = [b - a for a, b in zip(finished, finished[1:])]
    frame_time = _stats(frame_times)
    return {
        "frames": len(frames),
        "fps": 1000 / frame_time["mean"],
        "frame_time_ms": frame_time,
        "end_to_end_delay_ms": _stats([f["draw"][1] - f["capture"][1] for f in frames]),
        "steps_ms": {step: _stats([f[step][1] - f[step][0] for f in frames]) for step in STEPS},
    }


def _stats(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "p99": p99(values)}


def _report(summary: dict) -> str:
    rows = [("frame time", summary["frame_time_ms"]), ("end-to-end delay", summary["end_to_end_delay_ms"])]
    rows += [(f"  {step}", stats) for step, stats in summary["steps_ms"].items()]
    dropped = f", {summary['dropped_frames']} dropped by the camera" if "dropped_frames" in summary else ""
    lines = [f"{summary['frames']} frames, {summary['fps']:.1f} FPS{dropped}",
             f"{'':18}{'mean ms':>10}{'P99 ms':>10}"]
    lines += [f"{name:18}{stats['mean']:10.2f}{stats['p99']:10.2f}" for name, stats in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-timing", description=__doc__)
    parser.add_argument("log", help="timing log (.jsonl) written by a pipeline run")
    parser.add_argument("--out", help="results file to write (default: the log's name with .summary.json)")
    args = parser.parse_args(argv)

    summary = summarize(load_timings(args.log))
    dropped = load_dropped(args.log)
    if dropped is not None:  # webcam runs of the C++ pipeline count them
        summary["dropped_frames"] = dropped
    out = Path(args.out) if args.out else Path(args.log).with_suffix(".summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(_report(summary))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
