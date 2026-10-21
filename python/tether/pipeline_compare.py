"""tether-pipeline-compare: time the whole pipeline in Python and in C++ on the same video.

Runs `tether-demo --headless` and the C++ `tether-pipeline` (both headless, so
neither draws) on the same video file with the same model, one after the other,
summarizes both timing logs with tether-timing's summary, and saves both
summaries and the speedups to --out. The two timing logs are kept next to it.

Speedups are Python's time over C++'s: above 1 means C++ is faster.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tether.demo import DEFAULT_MODEL
from tether.formats import STEPS, load_timings
from tether.parity import DEFAULT_CPP_DIR, REPO, cpp_binary
from tether.timing import summarize


def _run(command: list, name: str, env: dict | None = None) -> None:
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise SystemExit(f"tether-pipeline-compare: error: {name} failed (exit {done.returncode}): "
                         f"{done.stderr.strip()}")


def run_python(video: str, model: str, log: Path) -> None:
    # The same tether package as this module, not whichever one is installed.
    path = os.pathsep.join(p for p in (str(REPO / "python"), os.environ.get("PYTHONPATH")) if p)
    _run([sys.executable, "-m", "tether.demo", video, "--model", model, "--headless", "--timing", log],
         "the Python pipeline", env={**os.environ, "PYTHONPATH": path})


def run_cpp(cpp_dir: Path, video: str, model: str, log: Path) -> None:
    binary = cpp_binary(cpp_dir, "tether-pipeline")
    if binary is None:
        raise SystemExit(f"tether-pipeline-compare: error: no tether-pipeline in {cpp_dir}; "
                         "build it with cpp/build.sh Release")
    _run([binary, video, "--model", model, "--timing", log], "tether-pipeline")


def compare(python: dict, cpp: dict) -> dict:
    """Python's time over C++'s for frame time, FPS, end-to-end delay and each step.

    A step C++ spends no time on (headless draw) has no speedup: None.
    """
    def ratio(py_ms: float, cpp_ms: float) -> float | None:
        return py_ms / cpp_ms if cpp_ms > 0 else None

    return {
        "fps": cpp["fps"] / python["fps"],
        "frame_time": ratio(python["frame_time_ms"]["mean"], cpp["frame_time_ms"]["mean"]),
        "end_to_end_delay": ratio(python["end_to_end_delay_ms"]["mean"], cpp["end_to_end_delay_ms"]["mean"]),
        "steps": {step: ratio(python["steps_ms"][step]["mean"], cpp["steps_ms"][step]["mean"]) for step in STEPS},
    }


def _report(result: dict) -> str:
    py, cpp, speedup = result["python"], result["cpp"], result["speedup"]
    lines = [f"{py['frames']} frames of {result['video']}",
             f"{'':18}{'Python ms':>11}{'C++ ms':>9}{'speedup':>9}"]
    rows = [("frame time", py["frame_time_ms"], cpp["frame_time_ms"], speedup["frame_time"]),
            ("end-to-end delay", py["end_to_end_delay_ms"], cpp["end_to_end_delay_ms"], speedup["end_to_end_delay"])]
    rows += [(f"  {s}", py["steps_ms"][s], cpp["steps_ms"][s], speedup["steps"][s]) for s in STEPS]
    for name, a, b, x in rows:
        lines.append(f"{name:18}{a['mean']:11.2f}{b['mean']:9.2f}{(f'{x:.1f}x' if x else '-'):>9}")
    lines.append(f"FPS {py['fps']:.1f} (Python) vs {cpp['fps']:.1f} (C++): {speedup['fps']:.1f}x")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-pipeline-compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file both pipelines run on")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"YOLO26 ONNX model (default {DEFAULT_MODEL})")
    parser.add_argument("--cpp-dir", type=Path, default=DEFAULT_CPP_DIR,
                        help="C++ build folder holding tether-pipeline (default: cpp/build)")
    parser.add_argument("--out", type=Path,
                        help="results file to write (default: results/pipeline-compare-<date>-<time>.json in the repo)")
    args = parser.parse_args(argv)

    out = args.out or REPO / "results" / f"pipeline-compare-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    python_log = out.with_name(f"{out.stem}.python-timing.jsonl")
    cpp_log = out.with_name(f"{out.stem}.cpp-timing.jsonl")
    run_python(args.video, args.model, python_log)
    run_cpp(args.cpp_dir, args.video, args.model, cpp_log)

    python, cpp = summarize(load_timings(python_log)), summarize(load_timings(cpp_log))
    if python["frames"] != cpp["frames"]:
        raise SystemExit(f"tether-pipeline-compare: error: Python ran {python['frames']} frames but C++ ran "
                         f"{cpp['frames']}; they should read the same video")
    result = {"video": args.video, "model": args.model, "python": python, "cpp": cpp,
              "speedup": compare(python, cpp)}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(_report(result))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
