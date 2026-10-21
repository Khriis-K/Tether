"""The whole-pipeline Python vs C++ comparison (#11), and tether-demo's headless mode it runs."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tether import demo, pipeline_compare
from tether.formats import load_timings, save_timings
from tether.pipeline_compare import compare, main
from tether.timing import summarize

STAND_IN_MODEL = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "onnx" / "fixed_736x1280.onnx"


def frames_on_disk(tmp_path, n):
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.zeros((1080, 1920, 3), dtype=np.uint8))
    return str(tmp_path / "f%d.png")


def test_headless_mode_tracks_and_logs_every_frame_without_a_window(tmp_path, monkeypatch):
    def no_window(*args):
        raise AssertionError("headless mode opened a window")
    monkeypatch.setattr(cv2, "imshow", no_window)
    log = tmp_path / "timing.jsonl"

    demo.main([frames_on_disk(tmp_path, 3), "--model", str(STAND_IN_MODEL), "--headless", "--timing", str(log)])

    assert len(load_timings(log)) == 3


def run(fps_ms, n=5, detect_ms=10.0):
    """A sequential timing log: each frame takes `fps_ms`, of which `detect_ms` is detection."""
    frames = []
    for i in range(n):
        t = i * fps_ms
        rest = fps_ms - detect_ms
        frames.append({"capture": (t, t + rest / 2), "prep": (t + rest / 2, t + rest / 2),
                       "detect": (t + rest / 2, t + rest / 2 + detect_ms),
                       "track": (t + rest / 2 + detect_ms, t + fps_ms), "draw": (t + fps_ms, t + fps_ms)})
    return frames


def test_speedups_are_python_time_over_cpp_time():
    python, cpp = summarize(run(40.0, detect_ms=20.0)), summarize(run(20.0, detect_ms=10.0))

    speedup = compare(python, cpp)

    assert speedup["fps"] == pytest.approx(2.0)
    assert speedup["frame_time"] == pytest.approx(2.0)
    assert speedup["steps"]["detect"] == pytest.approx(2.0)


def test_a_step_that_takes_no_time_in_cpp_has_no_speedup():
    # Headless, draw does nothing in either pipeline.
    speedup = compare(summarize(run(40.0)), summarize(run(20.0)))

    assert speedup["steps"]["draw"] is None
    assert speedup["steps"]["prep"] is None


class FakeRuns:
    """Stand-ins for running each pipeline: they write the given timing logs."""

    def __init__(self, python_frames, cpp_frames):
        self.python_frames, self.cpp_frames, self.calls = python_frames, cpp_frames, []

    def python(self, video, model, log):
        self.calls.append(("python", video, model))
        save_timings(log, self.python_frames)

    def cpp(self, cpp_dir, video, model, log):
        self.calls.append(("cpp", video, model))
        save_timings(log, self.cpp_frames)


@pytest.fixture
def fake_runs(monkeypatch):
    def install(fake):
        monkeypatch.setattr(pipeline_compare, "run_python", fake.python)
        monkeypatch.setattr(pipeline_compare, "run_cpp", fake.cpp)
        return fake
    return install


def test_main_runs_both_pipelines_on_the_same_video_and_saves_the_comparison(tmp_path, fake_runs, capsys):
    fake = fake_runs(FakeRuns(run(40.0), run(20.0)))
    out = tmp_path / "compare.json"

    assert main(["clip.mp4", "--model", "m.onnx", "--out", str(out)]) == 0

    assert fake.calls == [("python", "clip.mp4", "m.onnx"), ("cpp", "clip.mp4", "m.onnx")]
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["video"] == "clip.mp4"
    assert result["model"] == "m.onnx"
    assert result["python"] == summarize(run(40.0))
    assert result["cpp"] == summarize(run(20.0))
    assert result["speedup"]["fps"] == pytest.approx(2.0)
    assert "2.0x" in capsys.readouterr().out


def test_main_keeps_both_timing_logs_next_to_the_results(tmp_path, fake_runs):
    fake_runs(FakeRuns(run(40.0), run(20.0)))
    out = tmp_path / "compare.json"

    main(["clip.mp4", "--out", str(out)])

    assert load_timings(tmp_path / "compare.python-timing.jsonl") == run(40.0)
    assert load_timings(tmp_path / "compare.cpp-timing.jsonl") == run(20.0)


def test_main_fails_when_the_pipelines_saw_different_frame_counts(tmp_path, fake_runs):
    fake_runs(FakeRuns(run(40.0, n=5), run(20.0, n=4)))

    with pytest.raises(SystemExit, match="5 frames .* 4"):
        main(["clip.mp4", "--out", str(tmp_path / "c.json")])


def test_a_missing_cpp_binary_says_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit, match="build it with cpp/build.sh"):
        pipeline_compare.run_cpp(tmp_path, "clip.mp4", "m.onnx", tmp_path / "t.jsonl")
