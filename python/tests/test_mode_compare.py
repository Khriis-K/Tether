"""tether-mode-compare: the C++ pipeline, sequential against parallel, on the same video."""

import json

import pytest

from tether import mode_compare
from tether.formats import load_timings, save_timings
from tether.mode_compare import compare_modes, main
from tether.timing import summarize


def sequential_log(frame_ms, n=5, detect_ms=10.0):
    """One step after another: each frame takes `frame_ms`, of which `detect_ms` is detection."""
    frames = []
    for i in range(n):
        t = i * frame_ms
        rest = frame_ms - detect_ms
        frames.append({"capture": (t, t + rest), "prep": (t + rest, t + rest),
                       "detect": (t + rest, t + frame_ms), "track": (t + frame_ms, t + frame_ms),
                       "draw": (t + frame_ms, t + frame_ms)})
    return frames


def parallel_log(frame_ms, delay_ms, n=5, detect_ms=10.0):
    """Overlapped: a frame finishes every `frame_ms`, each `delay_ms` after it was captured."""
    frames = []
    for i in range(n):
        done = delay_ms + i * frame_ms
        captured = done - delay_ms
        frames.append({"capture": (captured - 1, captured), "prep": (captured, captured),
                       "detect": (done - detect_ms, done), "track": (done, done), "draw": (done, done)})
    return frames


def test_speedup_is_parallel_fps_over_sequential_and_delay_is_parallel_over_sequential():
    sequential = summarize(sequential_log(40.0))
    parallel = summarize(parallel_log(20.0, delay_ms=60.0, detect_ms=15.0))

    result = compare_modes(sequential, parallel)

    assert result["fps"] == pytest.approx(2.0)
    assert result["frame_time"] == pytest.approx(2.0)  # sequential's over parallel's
    # Sequential: capture end to draw end is only the 10 ms detect. Above 1: parallel waits longer.
    assert result["end_to_end_delay"] == pytest.approx(60.0 / 10.0)
    assert result["steps"]["detect"] == pytest.approx(1.5)  # parallel's over sequential's


def test_a_step_that_takes_no_time_in_sequential_has_no_ratio():
    result = compare_modes(summarize(sequential_log(40.0)), summarize(parallel_log(20.0, 60.0)))

    assert result["steps"]["draw"] is None


class FakeRuns:
    """Stands in for running tether-pipeline: writes the timing log for the mode asked for."""

    def __init__(self, logs):
        self.logs, self.calls = logs, []

    def run(self, cpp_dir, video, model, mode, log, max_frames):
        self.calls.append((mode, video, model, max_frames))
        save_timings(log, self.logs[mode])


@pytest.fixture
def fake_runs(monkeypatch):
    def install(logs):
        fake = FakeRuns(logs)
        monkeypatch.setattr(mode_compare, "run_mode", fake.run)
        return fake
    return install


LOGS = {"sequential": sequential_log(40.0), "parallel": parallel_log(20.0, 60.0)}


def test_main_alternates_which_mode_runs_first_each_round(tmp_path, fake_runs):
    fake = fake_runs(LOGS)

    assert main(["clip.mp4", "--model", "m.onnx", "--rounds", "3", "--out", str(tmp_path / "c.json")]) == 0

    assert [c[0] for c in fake.calls] == ["sequential", "parallel", "parallel", "sequential",
                                         "sequential", "parallel"]
    assert all(c[1:] == ("clip.mp4", "m.onnx", None) for c in fake.calls)


def test_main_saves_every_rounds_summaries_and_ratios(tmp_path, fake_runs, capsys):
    fake_runs(LOGS)
    out = tmp_path / "c.json"

    main(["clip.mp4", "--model", "m.onnx", "--rounds", "2", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["video"] == "clip.mp4"
    assert result["model"] == "m.onnx"
    assert len(result["rounds"]) == 2
    assert result["rounds"][1]["order"] == ["parallel", "sequential"]
    for r in result["rounds"]:
        assert r["sequential"] == summarize(LOGS["sequential"])
        assert r["parallel"] == summarize(LOGS["parallel"])
        assert r["parallel_vs_sequential"]["fps"] == pytest.approx(2.0)
    assert "2.00x" in capsys.readouterr().out


def test_main_keeps_every_timing_log_next_to_the_results(tmp_path, fake_runs):
    fake_runs(LOGS)

    main(["clip.mp4", "--rounds", "2", "--out", str(tmp_path / "c.json")])

    for i in (1, 2):
        assert load_timings(tmp_path / f"c.round{i}-sequential-timing.jsonl") == LOGS["sequential"]
        assert load_timings(tmp_path / f"c.round{i}-parallel-timing.jsonl") == LOGS["parallel"]


def test_main_passes_max_frames_on(tmp_path, fake_runs):
    fake = fake_runs(LOGS)

    main(["clip.mp4", "--max-frames", "300", "--out", str(tmp_path / "c.json")])

    assert {c[3] for c in fake.calls} == {300}


def test_main_fails_when_the_modes_ran_different_frame_counts(tmp_path, fake_runs):
    fake_runs({"sequential": sequential_log(40.0, n=5), "parallel": parallel_log(20.0, 60.0, n=4)})

    with pytest.raises(SystemExit, match="5 frames .* 4"):
        main(["clip.mp4", "--out", str(tmp_path / "c.json")])


def test_run_mode_asks_tether_pipeline_for_the_mode(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(mode_compare, "cpp_binary", lambda cpp_dir, name: tmp_path / name)
    monkeypatch.setattr(mode_compare, "_run", lambda command, name: commands.append([str(c) for c in command]))

    mode_compare.run_mode(tmp_path, "clip.mp4", "m.onnx", "parallel", tmp_path / "t.jsonl", None)
    mode_compare.run_mode(tmp_path, "clip.mp4", "m.onnx", "sequential", tmp_path / "t.jsonl", 50)

    binary = str(tmp_path / "tether-pipeline")
    log = str(tmp_path / "t.jsonl")
    assert commands == [
        [binary, "clip.mp4", "--model", "m.onnx", "--timing", log, "--pipeline", "parallel"],
        [binary, "clip.mp4", "--model", "m.onnx", "--timing", log, "--pipeline", "sequential", "--max-frames", "50"],
    ]


def test_a_missing_cpp_binary_says_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit, match="build it with cpp/build.sh"):
        mode_compare.run_mode(tmp_path, "clip.mp4", "m.onnx", "parallel", tmp_path / "t.jsonl", None)
