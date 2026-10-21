"""tether-runtime-compare --compare cuda-graph: TensorRT without and with its CUDA graph."""

import json

import pytest

from tether import runtime_compare
from tether.formats import Detection, save_detections, save_timings
from tether.runtime_compare import main
from tether.timing import summarize


def det(score=0.9):
    return Detection((100.0, 100.0, 200.0, 200.0), score, 0)


def timing_log(detect_ms, frame_ms, n=5):
    frames = []
    for i in range(n):
        t = i * frame_ms
        rest = frame_ms - detect_ms
        frames.append({"capture": (t, t + rest), "prep": (t + rest, t + rest),
                       "detect": (t + rest, t + frame_ms), "track": (t + frame_ms, t + frame_ms),
                       "draw": (t + frame_ms, t + frame_ms)})
    return frames


LOGS = {False: timing_log(5.0, 15.0), True: timing_log(2.5, 12.5)}  # keyed by "graph on"


class FakeRuns:
    """Stands in for tether-pipeline: the graph's on unless --no-cuda-graph is passed."""

    def __init__(self, detections):
        self.detections, self.calls = detections, []

    def run(self, cpp_dir, video, model, runtime, extra, max_frames):
        graph = "--no-cuda-graph" not in extra
        self.calls.append((runtime, graph, extra))
        if "--dump" in extra:
            save_detections(extra[extra.index("--dump") + 1], self.detections[graph])
        else:
            save_timings(extra[extra.index("--timing") + 1], LOGS[graph])


@pytest.fixture
def fake_runs(monkeypatch):
    def install(detections):
        fake = FakeRuns(detections)
        monkeypatch.setattr(runtime_compare, "run_pipeline", fake.run)
        return fake
    return install


SAME = {False: [[det()]], True: [[det()]]}


def test_both_sides_run_tensorrt_one_without_the_graph(tmp_path, fake_runs):
    fake = fake_runs(SAME)

    assert main(["clip.mp4", "--compare", "cuda-graph", "--rounds", "2", "--out", str(tmp_path / "g.json")]) == 0

    assert {c[0] for c in fake.calls} == {"tensorrt"}
    assert [c[1] for c in fake.calls] == [False, True, False, True, True, False]  # dumps, then alternating rounds


def test_timing_runs_use_the_pipeline_mode_asked_for(tmp_path, fake_runs):
    fake = fake_runs(SAME)

    main(["clip.mp4", "--compare", "cuda-graph", "--pipeline", "parallel", "--rounds", "1",
          "--out", str(tmp_path / "g.json")])

    timing = [c[2] for c in fake.calls if "--timing" in c[2]]
    assert len(timing) == 2
    assert all(extra[extra.index("--pipeline") + 1] == "parallel" for extra in timing)
    assert all("--pipeline" not in c[2] for c in fake.calls if "--dump" in c[2])


def test_results_name_the_sides_and_the_speedup_is_the_graphs(tmp_path, fake_runs):
    fake_runs(SAME)
    out = tmp_path / "g.json"

    main(["clip.mp4", "--compare", "cuda-graph", "--pipeline", "parallel", "--rounds", "1", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["compare"] == "cuda-graph"
    assert result["sides"] == ["no-cuda-graph", "cuda-graph"]
    assert result["pipeline"] == "parallel"
    r = result["rounds"][0]
    assert r["no-cuda-graph"] == summarize(LOGS[False])
    assert r["cuda-graph"] == summarize(LOGS[True])
    assert r["speedup"]["steps"]["detect"] == pytest.approx(2.0)
    assert (tmp_path / "g.cuda-graph-detections.jsonl").is_file()
    assert (tmp_path / "g.round1-no-cuda-graph-timing.jsonl").is_file()


def test_identical_dumps_are_reported_as_such(tmp_path, fake_runs):
    fake_runs(SAME)
    out = tmp_path / "g.json"

    main(["clip.mp4", "--compare", "cuda-graph", "--rounds", "1", "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["dumps_identical"] is True


def test_a_tiny_score_difference_agrees_but_isnt_identical(tmp_path, fake_runs, capsys):
    fake_runs({False: [[det(0.9)]], True: [[det(0.9001)]]})
    out = tmp_path / "g.json"

    assert main(["clip.mp4", "--compare", "cuda-graph", "--rounds", "1", "--out", str(out)]) == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["agrees"] is True
    assert result["dumps_identical"] is False
    assert "differ" in capsys.readouterr().out


def test_a_missing_detection_is_labelled_with_its_side(tmp_path, fake_runs):
    fake_runs({False: [[det()]], True: [[]]})
    out = tmp_path / "g.json"

    assert main(["clip.mp4", "--compare", "cuda-graph", "--rounds", "1", "--out", str(out)]) == 1

    missing = json.loads(out.read_text(encoding="utf-8"))["agreement"]["without_partner"]
    assert [m["runtime"] for m in missing] == ["no-cuda-graph"]
