"""tether-parity: check the C++ tracker against the Python one, and time both.

Runs each detection file (and, with --scripted, a fixed set of scripted scenes)
through both trackers at their defaults. It passes only when every frame has the
same tracks in the same order, with the same ID, status, class, missed-frame count
and path length, and every box, predicted box and path point within --tolerance
pixels; floating-point math differs slightly between the languages. On a mismatch
it names the first differing frame and track.

It also times the tracking step alone (no file reading or writing) in both
languages, as the median of --repeats runs over the whole file, and saves the
results and speedup to --out.
"""

import argparse
import json
import random
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from tether.formats import Detection, Track, load_detections, load_tracks, save_detections
from tether.tracker import Tracker

TOLERANCE = 1e-6  # pixels; the port differs by about 1e-9 on real files (docs/benchmarks/cpp-port-parity.md)
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CPP_DIR = REPO / "cpp" / "build"

Frames = list[list[Track]]


@dataclass(frozen=True)
class Mismatch:
    frame: int
    track_id: int | None  # None when the files differ in frame count
    what: str

    def __str__(self) -> str:
        where = f"frame {self.frame}" + (f", track {self.track_id}" if self.track_id is not None else "")
        return f"{where}: {self.what}"


def first_mismatch(python: Frames, cpp: Frames, tolerance: float = TOLERANCE) -> Mismatch | None:
    """The first frame and track where the two outputs differ, or None if they match."""
    for f, (py_tracks, cpp_tracks) in enumerate(zip(python, cpp)):
        for k in range(max(len(py_tracks), len(cpp_tracks))):
            if k >= len(cpp_tracks):
                return Mismatch(f, py_tracks[k].id, "only Python has this track")
            if k >= len(py_tracks):
                return Mismatch(f, cpp_tracks[k].id, "only C++ has this track")
            problem = _difference(py_tracks[k], cpp_tracks[k], tolerance)
            if problem:
                return Mismatch(f, py_tracks[k].id, problem)
    if len(python) != len(cpp):
        return Mismatch(min(len(python), len(cpp)), None,
                        f"Python has {len(python)} frames, C++ has {len(cpp)}")
    return None


def _difference(py: Track, cpp: Track, tolerance: float) -> str | None:
    if py.id != cpp.id:
        return f"C++ has track {cpp.id} here instead"
    for field in ("status", "cls", "frames_since_seen"):
        a, b = getattr(py, field), getattr(cpp, field)
        if a != b:
            return f"{field} {a} (Python) vs {b} (C++)"
    if len(py.path) != len(cpp.path):
        return f"path length {len(py.path)} (Python) vs {len(cpp.path)} (C++)"
    for field in ("box", "predicted_box"):
        a, b = getattr(py, field), getattr(cpp, field)
        if not _within(a, b, tolerance):
            return f"{field} {a} (Python) vs {b} (C++)"
    for a, b in zip(py.path, cpp.path):
        if not _within(a, b, tolerance):
            return f"path point {a} (Python) vs {b} (C++)"
    return None


def max_box_difference(python: Frames, cpp: Frames) -> float:
    """Largest coordinate difference over every box and predicted box both outputs have."""
    return max((_max_difference(getattr(a, field), getattr(b, field))
                for pf, cf in zip(python, cpp) for a, b in zip(pf, cf)
                for field in ("box", "predicted_box")), default=0.0)


def _within(a, b, tolerance: float) -> bool:
    # Written as <= so a NaN on either side fails: every comparison with NaN is False.
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def _max_difference(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def time_python(frames: list[list[Detection]], repeats: int) -> list[float]:
    """Milliseconds for a fresh default tracker over every frame, once per repeat."""
    times = []
    for _ in range(repeats):
        tracker = Tracker()
        start = time.perf_counter()
        for dets in frames:
            tracker.update(dets)
        times.append((time.perf_counter() - start) * 1000)
    return times


def _run_cpp(cpp_dir: Path, name: str, *args) -> str:
    """Run a C++ binary from `cpp_dir` (or its Release folder, for Visual Studio builds); returns its output."""
    candidates = [Path(cpp_dir) / sub / f"{name}{ext}" for sub in ("Release", ".") for ext in (".exe", "")]
    binary = next((c for c in candidates if c.is_file()), None)
    if binary is None:
        raise SystemExit(f"tether-parity: error: no {name} in {cpp_dir}; build it with cpp/build.sh Release")
    done = subprocess.run([binary, *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"tether-parity: error: {name} failed (exit {done.returncode}): {done.stderr.strip()}")
    return done.stdout


def run_cpp_tracker(cpp_dir: Path, detections_path: Path, tracks_path: Path) -> None:
    _run_cpp(cpp_dir, "tether-track", detections_path, tracks_path)


def time_cpp(cpp_dir: Path, detections_path: Path, repeats: int) -> list[float]:
    return json.loads(_run_cpp(cpp_dir, "tether-time-track", detections_path, "--repeats", str(repeats)))["repeats_ms"]


def scripted_inputs() -> dict[str, list[list[Detection]]]:
    """Fixed scenes covering the tracker's stages, by name."""
    def box(x, y=100.0, size=20.0):
        return (x, y, x + size, y + size)

    def det(b, score=0.9, cls=0):
        return Detection(box=b, score=score, cls=cls)

    return {
        "steady": [[det(box(10.0 + 4 * f))] for f in range(40)],
        "crossing": [[det(box(10.0 + 6 * f, 100.0)), det(box(250.0 - 6 * f, 104.0))] for f in range(40)],
        "disappear-and-return": [[det(box(10.0 + 5 * f))] if not 12 <= f < 20 else [] for f in range(35)],
        "removed-after-missing": [[det(box(10.0 + 5 * f))] if f < 6 else [] for f in range(45)],
        "low-score": [[det(box(10.0 + 5 * f), score=0.9 if f < 5 or f % 3 == 0 else 0.2)] for f in range(30)],
        "far-jump": [[det(box(10.0 + 5 * f if f < 10 else 400.0 + 5 * f))] for f in range(25)],
        "busy": _busy_scene(),
    }


def _busy_scene(n_frames: int = 60, seed: int = 0) -> list[list[Detection]]:
    """Two objects moving steadily plus random noise boxes, shuffled, like tests/test_cli.py's scene."""
    rng = random.Random(seed)
    frames = []
    for f in range(n_frames):
        dets = [Detection(box=(2.0 * f, 50.0, 2.0 * f + 20.0, 70.0), score=0.9, cls=0),
                Detection(box=(300.0 - 3.0 * f, 200.0, 330.0 - 3.0 * f, 240.0), score=0.8, cls=2)]
        for _ in range(rng.randint(0, 3)):
            x, y = rng.uniform(0, 600), rng.uniform(0, 400)
            dets.append(Detection(box=(x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40)),
                                  score=rng.uniform(0.1, 0.6), cls=rng.randint(0, 2)))
        rng.shuffle(dets)
        frames.append(dets)
    return frames


def check(name: str, detections_path: Path, cpp_tracks_path: Path, cpp_dir: Path, tolerance: float,
          repeats: int) -> dict:
    """Run, compare and time one detection file; the C++ tracks are written to `cpp_tracks_path`."""
    detections = load_detections(detections_path)
    tracker = Tracker()
    python = [tracker.update(d) for d in detections]
    run_cpp_tracker(cpp_dir, detections_path, cpp_tracks_path)
    cpp = load_tracks(cpp_tracks_path)

    mismatch = first_mismatch(python, cpp, tolerance)
    python_ms = statistics.median(time_python(detections, repeats))
    cpp_ms = statistics.median(time_cpp(cpp_dir, detections_path, repeats))
    return {
        "name": name,
        "frames": len(detections),
        "match": mismatch is None,
        "first_mismatch": asdict(mismatch) if mismatch else None,
        "max_box_difference": max_box_difference(python, cpp),
        "python_ms": python_ms,
        "cpp_ms": cpp_ms,
        "speedup": python_ms / cpp_ms,
    }


def _report(r: dict) -> str:
    verdict = "match" if r["match"] else f"MISMATCH at {Mismatch(**r['first_mismatch'])}"
    return (f"{r['name']}: {r['frames']} frames, {verdict}, largest box difference {r['max_box_difference']:.1e} px; "
            f"tracking step {r['python_ms']:.2f} ms Python, {r['cpp_ms']:.2f} ms C++, {r['speedup']:.1f}x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-parity", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("detections", nargs="*", help="detection files (.jsonl) to run through both trackers")
    parser.add_argument("--scripted", action="store_true", help="also run the built-in scripted scenes")
    parser.add_argument("--cpp-dir", type=Path, default=DEFAULT_CPP_DIR,
                        help="C++ build folder holding tether-track and tether-time-track (default: cpp/build)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help=f"largest allowed box, predicted box or path difference in pixels (default {TOLERANCE})")
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per file; the median is kept (default 5)")
    parser.add_argument("--out", type=Path,
                        help="results file to write (default: results/parity-<date>-<time>.json in the repo)")
    args = parser.parse_args(argv)
    if not args.detections and not args.scripted:
        parser.error("give detection files, --scripted, or both")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        inputs = [(Path(p).name, Path(p)) for p in args.detections]  # (name, detection file)
        if args.scripted:
            for name, frames in scripted_inputs().items():
                path = work_dir / f"{name}.jsonl"
                save_detections(path, frames)
                inputs.append((name, path))
        for i, (name, path) in enumerate(inputs):
            cpp_tracks_path = work_dir / f"cpp-tracks-{i}.jsonl"
            results.append(check(name, path, cpp_tracks_path, args.cpp_dir, args.tolerance, args.repeats))
            results[-1]["path"] = str(path) if i < len(args.detections) else None  # None for a scripted scene
            print(_report(results[-1]))

    total_python = sum(r["python_ms"] for r in results)
    total_cpp = sum(r["cpp_ms"] for r in results)
    summary = {
        "tolerance": args.tolerance,
        "repeats": args.repeats,
        "all_match": all(r["match"] for r in results),
        "speedup": total_python / total_cpp,  # over all inputs together, so long files weigh more
        "inputs": results,
    }
    out = args.out or REPO / "results" / f"parity-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'all match' if summary['all_match'] else 'MISMATCH'}; tracking step {summary['speedup']:.1f}x faster in C++")
    print(f"saved {out}")
    return 0 if summary["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
