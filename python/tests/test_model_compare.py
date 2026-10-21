"""tether-model-compare: several models on TensorRT, timed on the same video, with MOT17 scores beside them."""

import json

import pytest

from tether import model_compare
from tether.formats import STEPS, load_timings, save_timings
from tether.model_compare import busy_share, main, markdown_table, meets_targets, model_order, model_row
from tether.timing import summarize


def frames(frame_ms: float, detect_ms: float, n: int = 4) -> list[dict]:
    """A timing log whose frames finish every frame_ms, each spending detect_ms in detect."""
    out = []
    for i in range(n):
        t = i * frame_ms
        out.append({s: (t, t + (detect_ms if s == "detect" else 0.0)) for s in STEPS})
    return out


def summary(fps: float, p99_ms: float, detect_ms: float = 2.0, capture_busy: float = 0.99) -> dict:
    return {"fps": fps, "frame_time_ms": {"mean": 1000 / fps, "p99": p99_ms},
            "steps_ms": {"detect": {"mean": detect_ms, "p99": detect_ms}},
            "busy_share": {"capture": capture_busy, "prep": 0.7, "detect": 0.5, "track": 0.01, "draw": 0.0}}


MOT17 = {"combined": {"HOTA": 41.46, "IDF1": 49.42, "IDSW": 734, "Recall": 55.1, "Precision": 85.54}}


def test_model_order_rotates_so_each_model_takes_each_place():
    models = ["a", "b", "c"]

    assert [model_order(models, i) for i in (1, 2, 3, 4)] == [
        ["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"], ["a", "b", "c"]]


def test_targets_are_120_fps_or_more_and_p99_strictly_under_16_ms():
    assert meets_targets(summary(120.0, 15.99)) == {"fps": True, "p99": True, "both": True}
    assert meets_targets(summary(119.99, 15.0)) == {"fps": False, "p99": True, "both": False}
    assert meets_targets(summary(200.0, 16.0)) == {"fps": True, "p99": False, "both": False}


def test_busy_share_is_each_step_s_total_time_over_the_run_s_wall_time():
    # Two frames: capture 0-4 and 4-8, detect 4-7 and 8-11, the last draw ends at 12.
    frames = [{"capture": (0.0, 4.0), "prep": (4.0, 4.0), "detect": (4.0, 7.0), "track": (7.0, 7.5),
               "draw": (7.5, 8.0)},
              {"capture": (4.0, 8.0), "prep": (8.0, 8.0), "detect": (8.0, 11.0), "track": (11.0, 11.5),
               "draw": (11.5, 12.0)}]

    busy = busy_share(frames)

    assert busy == pytest.approx({"capture": 8 / 12, "prep": 0.0, "detect": 6 / 12, "track": 1 / 12,
                                  "draw": 1 / 12})


def test_row_gives_the_range_over_runs_and_the_shape_as_width_by_height():
    runs = [summary(130.0, 9.0, 4.0), summary(125.0, 11.0, 4.5)]

    row = model_row("models/yolo26s_1280x736.onnx", (736, 1280), runs, MOT17)

    assert row["input"] == "1280x736"
    assert row["runs"] == 2
    assert row["fps"] == [125.0, 130.0]
    assert row["frame_time_ms"] == pytest.approx([1000 / 130, 1000 / 125])
    assert row["frame_time_p99_ms"] == [9.0, 11.0]
    assert row["detect_ms"] == [4.0, 4.5]
    assert row["slowest_stage"] == "capture"
    assert row["slowest_busy_share"] == [0.99, 0.99]
    assert row["meets_targets"] is True
    assert row["mot17"] == {"HOTA": 41.46, "IDF1": 49.42, "IDSW": 734}


def test_row_meets_a_target_only_if_every_run_did():
    runs = [summary(130.0, 9.0), summary(118.0, 17.0)]

    row = model_row("m.onnx", (640, 640), runs, None)

    assert (row["meets_fps"], row["meets_p99"], row["meets_targets"]) == (False, False, False)
    assert row["mot17"] is None


def test_row_names_the_stage_busiest_over_all_runs():
    runs = [summary(130.0, 9.0, capture_busy=0.6), summary(125.0, 9.0, capture_busy=0.75)]

    row = model_row("m.onnx", (736, 1280), runs, None)

    assert row["slowest_stage"] == "prep"  # 0.7 + 0.7 beats capture's 0.6 + 0.75
    assert row["slowest_busy_share"] == [0.7, 0.7]


def test_table_shows_ranges_marks_and_scores_and_dashes_for_missing_scores():
    rows = [model_row("n.onnx", (640, 640), [summary(190.0, 7.0), summary(191.0, 7.0)], MOT17),
            model_row("m.onnx", (736, 1280), [summary(60.0, 20.0)], None)]

    table = markdown_table(rows).splitlines()

    assert table[0].startswith("| Model | Input (w×h) | FPS |")
    assert table[2] == ("| n | 640×640 | 190.00–191.00 | 5.24–5.26 | 7.00 | 2.00 | capture (99.0%) | yes | yes "
                        "| 41.46 | 49.42 | 734 |")
    assert table[3] == ("| m | 1280×736 | 60.00 | 16.67 | 20.00 | 2.00 | capture (99.0%) | no | no "
                        "| — | — | — |")


class FakeRuns:
    """Stands in for tether-pipeline: writes each model's canned timing log where the run asked."""

    def __init__(self, logs: dict):
        self.logs = logs
        self.calls = []

    def run(self, cpp_dir, video, model, mode, log, max_frames):
        self.calls.append((model, video, mode, max_frames))
        save_timings(log, self.logs[model])


@pytest.fixture
def fake_runs(monkeypatch):
    def install(logs):
        fake = FakeRuns(logs)
        monkeypatch.setattr(model_compare, "run_model", fake.run)
        monkeypatch.setattr(model_compare, "_input_size", lambda m: (640, 640) if m == "n.onnx" else (736, 1280))
        return fake
    return install


LOGS = {"n.onnx": frames(5.0, 2.0), "s_1280x736.onnx": frames(10.0, 6.0)}


def test_main_rotates_the_model_order_each_round_in_the_mode_asked(tmp_path, fake_runs):
    fake = fake_runs(LOGS)

    main(["clip.mp4", "--models", "n.onnx", "s_1280x736.onnx", "--rounds", "3", "--pipeline", "sequential",
          "--max-frames", "50", "--out", str(tmp_path / "c.json")])

    assert [c[0] for c in fake.calls] == ["n.onnx", "s_1280x736.onnx", "s_1280x736.onnx", "n.onnx",
                                          "n.onnx", "s_1280x736.onnx"]
    assert all(c[1:] == ("clip.mp4", "sequential", 50) for c in fake.calls)


def test_main_saves_every_run_and_a_row_per_model_with_its_mot17_scores(tmp_path, fake_runs, capsys):
    fake_runs(LOGS)
    acc = tmp_path / "trackers" / "tether_n"
    acc.mkdir(parents=True)
    (acc / "results.json").write_text(json.dumps(MOT17), encoding="utf-8")
    out = tmp_path / "c.json"

    assert main(["clip.mp4", "--models", "n.onnx", "s_1280x736.onnx", "--rounds", "2",
                 "--accuracy-dir", str(tmp_path / "trackers"), "--out", str(out)]) == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["pipeline"] == "parallel"
    assert result["targets"] == {"fps": 120.0, "p99_ms": 16.0}
    n_log = LOGS["n.onnx"]
    assert result["runs"]["n.onnx"] == [{**summarize(n_log), "busy_share": pytest.approx(busy_share(n_log))}] * 2
    n, s = result["rows"]
    assert (n["input"], n["meets_targets"], n["mot17"]["HOTA"]) == ("640x640", True, 41.46)
    assert (s["input"], s["meets_fps"], s["mot17"]) == ("1280x736", False, None)
    assert load_timings(tmp_path / "c.round2-s_1280x736-timing.jsonl") == LOGS["s_1280x736.onnx"]
    assert "| n | 640×640 | 200.00 |" in capsys.readouterr().out


def test_main_rejects_zero_rounds(tmp_path, fake_runs):
    fake_runs(LOGS)

    with pytest.raises(SystemExit):
        main(["clip.mp4", "--models", "n.onnx", "--rounds", "0", "--out", str(tmp_path / "c.json")])
