"""tether-runtime-compare --compare frame-prep: TensorRT with CPU frame prep against CUDA frame prep."""

import json

import pytest

from tether import runtime_compare
from tether.formats import Detection, save_detections, save_timings
from tether.runtime_compare import main
from tether.timing import summarize


def det(score=0.9):
    return Detection((100.0, 100.0, 200.0, 200.0), score, 0)


def timing_log(prep_ms, detect_ms, frame_ms, n=5):
    frames = []
    for i in range(n):
        t = i * frame_ms
        capture = frame_ms - prep_ms - detect_ms
        frames.append({"capture": (t, t + capture), "prep": (t + capture, t + capture + prep_ms),
                       "detect": (t + capture + prep_ms, t + frame_ms), "track": (t + frame_ms, t + frame_ms),
                       "draw": (t + frame_ms, t + frame_ms)})
    return frames


LOGS = {"cpu": timing_log(8.0, 5.0, 16.0), "cuda": timing_log(1.0, 4.0, 8.0)}  # keyed by --prep


class FakeRuns:
    """Stands in for tether-pipeline: CPU prep unless --prep cuda is passed."""

    def __init__(self, detections):
        self.detections, self.calls = detections, []

    def run(self, cpp_dir, video, model, runtime, extra, max_frames):
        prep = extra[extra.index("--prep") + 1] if "--prep" in extra else "cpu"
        self.calls.append((runtime, prep, extra))
        if "--dump" in extra:
            save_detections(extra[extra.index("--dump") + 1], self.detections[prep])
        else:
            save_timings(extra[extra.index("--timing") + 1], LOGS[prep])


@pytest.fixture
def fake_runs(monkeypatch):
    def install(detections):
        fake = FakeRuns(detections)
        monkeypatch.setattr(runtime_compare, "run_pipeline", fake.run)
        return fake
    return install


SAME = {"cpu": [[det()]], "cuda": [[det()]]}


def test_both_sides_run_tensorrt_one_with_cuda_prep(tmp_path, fake_runs):
    fake = fake_runs(SAME)

    assert main(["clip.mp4", "--compare", "frame-prep", "--rounds", "2", "--out", str(tmp_path / "p.json")]) == 0

    assert {c[0] for c in fake.calls} == {"tensorrt"}
    assert [c[1] for c in fake.calls] == ["cpu", "cuda", "cpu", "cuda", "cuda", "cpu"]  # dumps, then alternating


def test_results_name_the_sides_and_the_prep_speedup_is_cudas(tmp_path, fake_runs):
    fake_runs(SAME)
    out = tmp_path / "p.json"

    main(["clip.mp4", "--compare", "frame-prep", "--pipeline", "parallel", "--rounds", "1", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["compare"] == "frame-prep"
    assert result["sides"] == ["cpu-prep", "cuda-prep"]
    r = result["rounds"][0]
    assert r["cpu-prep"] == summarize(LOGS["cpu"])
    assert r["cuda-prep"] == summarize(LOGS["cuda"])
    assert r["speedup"]["steps"]["prep"] == pytest.approx(8.0)
    assert r["speedup"]["fps"] == pytest.approx(2.0)


def test_the_report_gives_the_prep_speedup_and_the_frame_time_saved(tmp_path, fake_runs, capsys):
    fake_runs(SAME)

    main(["clip.mp4", "--compare", "frame-prep", "--rounds", "1", "--out", str(tmp_path / "p.json")])

    report = capsys.readouterr().out
    assert "prep 8.00 -> 1.00 ms (8.00x)" in report
    assert "frame time 16.00 -> 8.00 ms (8.00 ms saved)" in report
