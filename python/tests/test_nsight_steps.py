"""tools/nsight_steps.py: where a profiled run's time goes, from an Nsight Systems export (#15)."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "nsight_steps", Path(__file__).resolve().parents[2] / "tools" / "nsight_steps.py")
nsight_steps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nsight_steps)

# One TensorRT detect call, as the profiler sees it: copy in, launch the kernels,
# copy out, wait for the GPU, then decode on the CPU. Times in ms.
DETECT = (0.0, 100.0)
CALLS = [
    nsight_steps.Call("cudaMemcpyAsync_v3020", 2.0, 10.0, 1),
    nsight_steps.Call("cuLaunchKernelEx", 11.0, 30.0, 2),
    nsight_steps.Call("cuLaunchKernelEx", 31.0, 59.0, 3),
    nsight_steps.Call("cudaMemcpyAsync_v3020", 60.0, 61.0, 4),
    nsight_steps.Call("cudaStreamSynchronize_v3020", 61.0, 95.0, 5),
]
GPU = {1: (5.0, 12.0), 2: (20.0, 40.0), 3: (45.0, 85.0), 4: (85.0, 86.0)}


def test_detect_splits_into_copy_in_launch_copy_out_wait_and_the_rest():
    parts = nsight_steps.split_detect(DETECT, CALLS, GPU)
    assert parts["total"] == 100.0
    assert parts["h2d_call"] == 8.0
    assert parts["launch"] == 50.0  # from the copy in returning to the copy out being called
    assert parts["d2h_call"] == 1.0
    assert parts["sync"] == 34.0
    assert parts["other"] == 7.0  # before the copy in, and decoding after the wait
    assert sum(parts[p] for p in nsight_steps.CPU_PARTS) == parts["total"]


def test_gpu_side_of_a_detect_call():
    parts = nsight_steps.split_detect(DETECT, CALLS, GPU)
    assert parts["gpu_h2d"] == 7.0
    assert parts["gpu_kernels"] == 60.0
    assert parts["gpu_d2h"] == 1.0
    # The GPU starts the copy in at 5 and finishes the copy out at 86, busy 68 of that.
    assert parts["gpu_span"] == 81.0
    assert parts["gpu_idle"] == 13.0
    assert parts["launches"] == 2


def test_calls_outside_the_detect_call_are_left_out():
    before = nsight_steps.Call("cudaMemcpyAsync_v3020", -5.0, -1.0, 9)
    after = nsight_steps.Call("cudaStreamSynchronize_v3020", 101.0, 120.0, 10)
    assert (nsight_steps.split_detect(DETECT, [before, *CALLS, after], GPU)
            == nsight_steps.split_detect(DETECT, CALLS, GPU))


def test_a_detect_call_without_cuda_is_all_other():
    parts = nsight_steps.split_detect((0.0, 33.0), [], {})
    assert parts["other"] == 33.0
    assert parts["h2d_call"] == parts["launch"] == parts["sync"] == parts["gpu_span"] == 0.0


def test_busy_share_is_each_step_s_time_over_the_run_s_wall_time():
    ranges = {"capture": [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)], "detect": [(6.0, 16.0), (16.0, 26.0), (26.0, 40.0)],
              "track": []}
    busy = nsight_steps.busy_share(ranges)
    assert busy["capture"] == pytest.approx(15.0 / 40.0)
    assert busy["detect"] == pytest.approx(34.0 / 40.0)
    assert "track" not in busy  # a step with no ranges has no share


def test_slowest_one_percent_shows_which_part_grew():
    fast = {"total": 6.0, "launch": 4.0, "sync": 1.0}
    slow = {"total": 12.0, "launch": 4.0, "sync": 7.0}
    parts = [fast] * 99 + [slow] * 2
    tail = nsight_steps.slowest(parts, ["launch", "sync"])
    assert tail["frames"] == 2
    assert tail["launch"] == 4.0 and tail["sync"] == 7.0


def _report(path: Path) -> None:
    """A tiny export with nsys's table layout: one detect call on one thread, plus a prep call."""
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER,
                                                  correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER, copyKind INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaMemcpyAsync_v3020'), (2, 'cuLaunchKernelEx'),
                                     (3, 'cudaStreamSynchronize_v3020');
        INSERT INTO NVTX_EVENTS VALUES (0, 4000000, 'prep', 7), (4000000, 10000000, 'detect', 8);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (4000000, 5000000, 8, 1, 1), (5000000, 6000000, 8, 2, 2), (6000000, 6100000, 8, 3, 1),
            (6100000, 9000000, 8, 4, 3), (5000000, 6000000, 7, 5, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (4500000, 5000000, 1, 1), (8000000, 8100000, 3, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (6000000, 8000000, 2);
    """)
    db.commit()
    db.close()


def test_reads_the_steps_and_cuda_calls_from_an_export(tmp_path):
    _report(tmp_path / "run.sqlite")
    result = nsight_steps.analyze(tmp_path / "run.sqlite")
    assert result["steps_ms"]["prep"]["mean"] == 4.0
    assert result["steps_ms"]["detect"]["mean"] == 6.0
    detect = result["detect_parts_ms"]
    assert detect["h2d_call"]["mean"] == pytest.approx(1.0)
    assert detect["launch"]["mean"] == pytest.approx(1.0)
    assert detect["sync"]["mean"] == pytest.approx(2.9)
    assert detect["gpu_kernels"]["mean"] == pytest.approx(2.0)
    assert detect["gpu_h2d"]["mean"] == pytest.approx(0.5)
    assert result["kernel_launches_per_detect"]["mean"] == 1  # prep's launch, on another thread, isn't counted
    assert result["busy_share"]["detect"] == pytest.approx(0.6)


def test_a_run_without_cuda_has_steps_but_no_detect_split(tmp_path):
    db = sqlite3.connect(tmp_path / "ort.sqlite")
    db.executescript("""
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        INSERT INTO NVTX_EVENTS VALUES (0, 33000000, 'detect', 8);
    """)
    db.commit()
    db.close()
    result = nsight_steps.analyze(tmp_path / "ort.sqlite")
    assert result["steps_ms"]["detect"]["mean"] == 33.0
    assert "detect_parts_ms" not in result
