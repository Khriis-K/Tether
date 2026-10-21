"""Where a profiled tether-pipeline run's time goes, from an Nsight Systems export: the numbers for #15.

    nsys profile -t cuda,nvtx --nvtx-domain-include=default --sample=none --cpuctxsw=none -o RUN \\
        tether-pipeline VIDEO --runtime tensorrt --pipeline parallel
    nsys export --type sqlite -o RUN.sqlite RUN.nsys-rep
    python tools/nsight_steps.py RUN.sqlite --out RUN-steps.json

Reads the pipeline's own NVTX ranges (capture, prep, detect, track, draw; see
`StepRange` in cpp/src/tether/pipeline.cpp) and, inside each detect range, the
CUDA calls made on the detect thread and the GPU work they started. Each
TensorRT detect call is split, on the CPU, into the copy to the GPU, launching
the engine's kernels (enqueueV3, or one cudaGraphLaunch when the engine is
replayed from a CUDA graph, #40), the copy back, waiting for the GPU, and the
rest (decoding the output); and on the GPU into the two copies, the kernels,
and the time the GPU sat idle between the first copy starting and the last
one ending. Times are milliseconds; percentiles are nearest-rank, as in
tether-timing. Profile a CUDA graph run with `--cuda-graph-trace=node`, or nsys
records the graph as a whole and its kernels can't be counted.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

from tether.formats import STEPS
from tether.timing import p99

CPU_PARTS = ("h2d_call", "launch", "d2h_call", "sync", "other")
GPU_PARTS = ("gpu_h2d", "gpu_kernels", "gpu_d2h", "gpu_span", "gpu_idle")
NS_PER_MS = 1e6


class Call(NamedTuple):
    """One CUDA runtime or driver call on the CPU, and the id that links it to its GPU work."""
    name: str
    start: float
    end: float
    correlation: int


def _named(call: Call, prefix: str) -> bool:
    """nsys names runtime calls with a version suffix (cudaMemcpyAsync_v3020), so match the prefix."""
    return call.name.startswith(prefix)


Span = tuple[float, float]


def _spans(work: Span | list[Span]) -> list[Span]:
    """A call's GPU work: one span, or several when one graph launch ran many kernels."""
    return work if isinstance(work, list) else [work]


def split_detect(detect: tuple[float, float], calls: list[Call], gpu: dict[int, Span | list[Span]]) -> dict:
    """One detect call's time, by part. `calls` are the detect thread's CUDA calls; `gpu` maps a
    call's correlation id to when its GPU work (a copy, a kernel, or a graph's kernels) ran."""
    start, end = detect
    calls = sorted((c for c in calls if start <= c.start and c.end <= end), key=lambda c: c.start)
    copies = [c for c in calls if _named(c, "cudaMemcpyAsync")]
    parts = dict.fromkeys(CPU_PARTS + GPU_PARTS, 0.0)
    parts["total"] = end - start
    if copies:
        h2d = copies[0]
        parts["h2d_call"] = h2d.end - h2d.start
        if len(copies) > 1:
            d2h = copies[-1]
            parts["d2h_call"] = d2h.end - d2h.start
            parts["launch"] = d2h.start - h2d.end
    parts["sync"] = sum(c.end - c.start for c in calls if _named(c, "cudaStreamSynchronize"))
    parts["other"] = parts["total"] - sum(parts[p] for p in CPU_PARTS if p != "other")

    def work(of: list[Call]) -> list[Span]:
        return [span for c in of if c.correlation in gpu for span in _spans(gpu[c.correlation])]

    def gpu_time(of: list[Call]) -> float:
        return sum(e - s for s, e in work(of))

    if copies:
        parts["gpu_h2d"] = gpu_time(copies[:1])
        parts["gpu_d2h"] = gpu_time(copies[1:])
    launches = [c for c in calls if _named(c, "cuLaunchKernel")]
    graphs = [c for c in calls if _named(c, "cudaGraphLaunch")]
    parts["launches"] = len(launches)
    parts["graph_launches"] = len(graphs)
    parts["kernels"] = len(work(launches + graphs))
    parts["gpu_kernels"] = gpu_time(launches + graphs)
    spans = work(calls)
    if spans:
        parts["gpu_span"] = max(e for _, e in spans) - min(s for s, _ in spans)
        parts["gpu_idle"] = parts["gpu_span"] - (parts["gpu_h2d"] + parts["gpu_kernels"] + parts["gpu_d2h"])
    return parts


def busy_share(ranges: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    """Each step's total time over the run's wall time (first step's start to last step's end).
    In a parallel run each step has its own thread, so the slowest stage is the one near 1."""
    spans = [span for r in ranges.values() for span in r]
    wall = max(e for _, e in spans) - min(s for s, _ in spans)
    return {step: sum(e - s for s, e in r) / wall for step, r in ranges.items() if r}


def slowest(parts: list[dict], names: list[str]) -> dict:
    """The mean of each part over the detect calls at or above the P99 total: what makes the slow ones slow."""
    cut = p99([p["total"] for p in parts])
    tail = [p for p in parts if p["total"] >= cut]
    return {"frames": len(tail), "total": sum(p["total"] for p in tail) / len(tail),
            **{n: sum(p[n] for p in tail) / len(tail) for n in names}}


def _stats(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "p99": p99(values)}


def analyze(path: Path) -> dict:
    """A capture's per-step times and busy shares and, if it ran TensorRT, each detect call split by part."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ranges = db.execute(
        f"SELECT text, start, end, globalTid FROM NVTX_EVENTS WHERE text IN ({','.join('?' * len(STEPS))}) "
        "ORDER BY start", STEPS).fetchall()
    by_step: dict[str, list[tuple[float, float, int]]] = {s: [] for s in STEPS}
    for text, start, end, tid in ranges:
        by_step[text].append((start / NS_PER_MS, end / NS_PER_MS, tid))

    calls: dict[int, list[Call]] = {}
    gpu: dict[int, list[Span]] = {}  # a graph launch's kernels all carry its correlation id
    # A run without CUDA (ONNX Runtime) has none of nsys's CUDA tables.
    tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        for name, start, end, tid, corr in db.execute(
                "SELECT s.value, r.start, r.end, r.globalTid, r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id = r.nameId"):
            calls.setdefault(tid, []).append(Call(name, start / NS_PER_MS, end / NS_PER_MS, corr))
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if table in tables:
            for start, end, corr in db.execute(f"SELECT start, end, correlationId FROM {table}"):
                gpu.setdefault(corr, []).append((start / NS_PER_MS, end / NS_PER_MS))
    db.close()

    result = {"report": str(path),
              "steps_ms": {s: _stats([e - b for b, e, _ in r]) for s, r in by_step.items() if r},
              "ranges": {s: len(r) for s, r in by_step.items()}}
    result["busy_share"] = busy_share({s: [(b, e) for b, e, _ in r] for s, r in by_step.items()})

    detects = [split_detect((b, e), calls.get(tid, []), gpu) for b, e, tid in by_step["detect"]]
    if any(d["h2d_call"] for d in detects):
        names = list(CPU_PARTS + GPU_PARTS)
        result["detect_parts_ms"] = {n: _stats([d[n] for d in detects]) for n in ["total", *names]}
        result["detect_slowest_1pct_ms"] = slowest(detects, names)
        result["kernel_launches_per_detect"] = _stats([d["launches"] for d in detects])
        result["graph_launches_per_detect"] = _stats([d["graph_launches"] for d in detects])
        result["kernels_per_detect"] = _stats([d["kernels"] for d in detects])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path, help="an nsys export (.sqlite) of a tether-pipeline run")
    parser.add_argument("--out", type=Path, help="write the result here as JSON (default: print it)")
    args = parser.parse_args(argv)
    result = analyze(args.report)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
