"""tools/nsight_steps.py on a detect call that replays a CUDA graph (#40): one cudaGraphLaunch on
the CPU, and many kernels on the GPU that all carry that launch's correlation id."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "nsight_steps", Path(__file__).resolve().parents[2] / "tools" / "nsight_steps.py")
nsight_steps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nsight_steps)

DETECT = (0.0, 20.0)
CALLS = [
    nsight_steps.Call("cudaMemcpyAsync_v3020", 1.0, 3.0, 1),
    nsight_steps.Call("cudaGraphLaunch_v10000", 3.0, 3.5, 2),
    nsight_steps.Call("cudaMemcpyAsync_v3020", 3.5, 4.0, 3),
    nsight_steps.Call("cudaStreamSynchronize_v3020", 4.0, 18.0, 4),
]
# Three kernels from the one graph launch, with a 1 ms gap between the second and third.
GPU = {1: [(2.0, 4.0)], 2: [(4.0, 6.0), (6.0, 9.0), (10.0, 12.0)], 3: [(12.0, 13.0)]}


def test_a_graph_launch_counts_as_the_launch_and_its_kernels_add_up():
    parts = nsight_steps.split_detect(DETECT, CALLS, GPU)

    assert parts["launch"] == 0.5  # the one cudaGraphLaunch, between the two copies
    assert parts["graph_launches"] == 1
    assert parts["launches"] == 0  # no kernels launched one by one
    assert parts["kernels"] == 3
    assert parts["gpu_kernels"] == 7.0
    assert parts["gpu_span"] == 11.0  # first copy starts at 2, copy back ends at 13
    assert parts["gpu_idle"] == 1.0


def test_one_span_per_call_still_works():
    parts = nsight_steps.split_detect(DETECT, CALLS, {1: (2.0, 4.0), 2: (4.0, 9.0), 3: (12.0, 13.0)})

    assert parts["kernels"] == 1
    assert parts["gpu_kernels"] == 5.0


def test_an_export_with_graph_kernels_keeps_every_one(tmp_path):
    path = tmp_path / "graph.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER,
                                                  correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, correlationId INTEGER, copyKind INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaMemcpyAsync_v3020'), (2, 'cudaGraphLaunch_v10000'),
                                     (3, 'cudaStreamSynchronize_v3020');
        INSERT INTO NVTX_EVENTS VALUES (0, 10000000, 'detect', 8);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (1000000, 2000000, 8, 1, 1), (2000000, 2100000, 8, 2, 2), (2100000, 2200000, 8, 3, 1),
            (2200000, 9000000, 8, 4, 3);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1500000, 2500000, 1, 1), (7000000, 7100000, 3, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (2500000, 4000000, 2), (4000000, 5000000, 2), (5000000, 7000000, 2);
    """)
    db.commit()
    db.close()

    result = nsight_steps.analyze(path)

    assert result["kernels_per_detect"]["mean"] == 3
    assert result["graph_launches_per_detect"]["mean"] == 1
    assert result["kernel_launches_per_detect"]["mean"] == 0
    assert result["detect_parts_ms"]["gpu_kernels"]["mean"] == pytest.approx(4.5)
    assert result["detect_parts_ms"]["launch"]["mean"] == pytest.approx(0.1)
    assert result["detect_parts_ms"]["gpu_idle"]["mean"] == pytest.approx(0.0)
