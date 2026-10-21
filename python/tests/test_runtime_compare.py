"""tether-runtime-compare: the C++ pipeline on ONNX Runtime against TensorRT."""

import json

import pytest

from tether import runtime_compare
from tether.formats import Detection, load_timings, save_detections, save_timings
from tether.runtime_compare import agreement, main, speedup
from tether.timing import summarize

BOX = (100.0, 100.0, 200.0, 200.0)


def det(box=BOX, score=0.9, cls=0):
    return Detection(box, score, cls)


def test_identical_detections_agree_completely():
    frames = [[det(), det((300.0, 300.0, 350.0, 380.0), 0.5, 2)], [det()]]

    a = agreement(frames, frames)

    assert a["checked"] == 6  # three detections, each checked from both sides
    assert a["partnered"] == 6
    assert a["agreement"] == 1.0
    assert a["worst_partnered_score_diff"] == 0.0
    assert a["worst_partnered_box_diff_px"] == 0.0
    assert a["without_partner"] == []


def test_small_box_and_score_differences_still_partner():
    nudged = det((100.5, 100.0, 200.0, 200.5), 0.89)

    a = agreement([[det()]], [[nudged]])

    assert a["agreement"] == 1.0
    assert a["worst_partnered_score_diff"] == pytest.approx(0.01)
    assert a["worst_partnered_box_diff_px"] == pytest.approx(0.5)


def test_a_detection_only_one_runtime_finds_has_no_partner():
    a = agreement([[det(), det((400.0, 400.0, 450.0, 450.0))]], [[det()]])

    assert a["checked"] == 3
    assert a["partnered"] == 2
    assert a["agreement"] == pytest.approx(2 / 3)
    assert a["worst_iou"] == 0.0
    assert a["without_partner"] == [{"frame": 0, "runtime": "onnxruntime", "box": [400.0, 400.0, 450.0, 450.0],
                                     "score": 0.9, "cls": 0, "best_iou": 0.0, "score_diff": None}]


def test_too_little_overlap_too_different_a_score_or_another_class_is_no_partner():
    shifted = det((110.0, 100.0, 210.0, 200.0))  # IoU 0.82
    rescored = det(score=0.85)
    other_class = det(cls=1)

    for other in (shifted, rescored, other_class):
        assert agreement([[det()]], [[other]])["partnered"] == 0, other


def test_of_two_near_identical_boxes_the_partner_is_the_one_with_the_closer_score():
    # ONNX Runtime put two boxes on one object; TensorRT's score matches the second, slightly offset one.
    twin = det((100.0, 100.0, 200.0, 201.0), 0.6)
    a = agreement([[det(score=0.9), twin]], [[det((100.0, 100.0, 200.0, 200.2), 0.6)]])

    # TensorRT's 0.6 overlaps the 0.9 box more but partners the 0.6 twin; the 0.9 box has none within 0.02.
    assert a["checked"] == 3
    assert a["partnered"] == 2
    assert a["worst_partnered_score_diff"] == pytest.approx(0.0)


def test_detections_under_the_tracker_threshold_arent_checked_but_can_be_partners():
    # 0.26 in one runtime, 0.245 in the other: the 0.26 one counts and finds its partner.
    a = agreement([[det(score=0.26), det((400.0, 400.0, 450.0, 450.0), 0.2)]], [[det(score=0.245)]])

    assert a["checked"] == 1
    assert a["partnered"] == 1


def test_no_detections_to_check_counts_as_agreement():
    assert agreement([[], []], [[], []])["agreement"] == 1.0


def test_runtimes_with_different_frame_counts_cant_be_compared():
    with pytest.raises(ValueError, match="2 frames .* 1"):
        agreement([[], []], [[]])


def timing_log(detect_ms, frame_ms, n=5):
    """One step after another: each frame takes `frame_ms`, of which `detect_ms` is detection."""
    frames = []
    for i in range(n):
        t = i * frame_ms
        rest = frame_ms - detect_ms
        frames.append({"capture": (t, t + rest), "prep": (t + rest, t + rest),
                       "detect": (t + rest, t + frame_ms), "track": (t + frame_ms, t + frame_ms),
                       "draw": (t + frame_ms, t + frame_ms)})
    return frames


LOGS = {"onnxruntime": timing_log(30.0, 40.0), "tensorrt": timing_log(5.0, 15.0)}


def test_speedup_is_onnx_runtimes_time_over_tensorrts():
    x = speedup(summarize(LOGS["onnxruntime"]), summarize(LOGS["tensorrt"]))

    assert x["steps"]["detect"] == pytest.approx(6.0)
    assert x["frame_time"] == pytest.approx(40.0 / 15.0)
    assert x["fps"] == pytest.approx(40.0 / 15.0)
    assert x["steps"]["draw"] is None  # no time in either


class FakeRuns:
    """Stands in for tether-pipeline: writes the dump or timing log asked for."""

    def __init__(self, detections, logs):
        self.detections, self.logs, self.calls = detections, logs, []

    def run(self, cpp_dir, video, model, runtime, extra, max_frames):
        self.calls.append((runtime, extra[0], video, model, max_frames))
        if extra[0] == "--dump":
            save_detections(extra[1], self.detections[runtime])
        else:
            save_timings(extra[1], self.logs[runtime])


@pytest.fixture
def fake_runs(monkeypatch):
    def install(detections, logs=LOGS):
        fake = FakeRuns(detections, logs)
        monkeypatch.setattr(runtime_compare, "run_pipeline", fake.run)
        return fake
    return install


SAME = {"onnxruntime": [[det()]], "tensorrt": [[det()]]}


def test_main_dumps_both_then_alternates_which_runtime_times_first(tmp_path, fake_runs):
    fake = fake_runs(SAME)

    assert main(["clip.mp4", "--model", "m.onnx", "--rounds", "3", "--out", str(tmp_path / "c.json")]) == 0

    assert [c[:2] for c in fake.calls] == [
        ("onnxruntime", "--dump"), ("tensorrt", "--dump"),
        ("onnxruntime", "--timing"), ("tensorrt", "--timing"),
        ("tensorrt", "--timing"), ("onnxruntime", "--timing"),
        ("onnxruntime", "--timing"), ("tensorrt", "--timing")]
    assert all(c[2:] == ("clip.mp4", "m.onnx", None) for c in fake.calls)


def test_main_saves_the_tolerance_agreement_and_every_rounds_timing(tmp_path, fake_runs, capsys):
    fake_runs(SAME)
    out = tmp_path / "c.json"

    main(["clip.mp4", "--model", "m.onnx", "--rounds", "2", "--out", str(out)])

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["tolerance"] == {"min_score": 0.25, "min_iou": 0.95, "max_score_diff": 0.02, "min_agreement": 0.99}
    assert result["agreement"]["partnered"] == 2
    assert result["agrees"] is True
    assert len(result["rounds"]) == 2
    for r in result["rounds"]:
        assert r["onnxruntime"] == summarize(LOGS["onnxruntime"])
        assert r["tensorrt"] == summarize(LOGS["tensorrt"])
        assert r["speedup"]["steps"]["detect"] == pytest.approx(6.0)
    for i in (1, 2):
        assert load_timings(tmp_path / f"c.round{i}-tensorrt-timing.jsonl") == LOGS["tensorrt"]
    assert (tmp_path / "c.onnxruntime-detections.jsonl").is_file()
    report = capsys.readouterr().out
    assert "PASS" in report
    assert "6.00x" in report


def test_main_fails_when_the_runtimes_disagree(tmp_path, fake_runs, capsys):
    fake_runs({"onnxruntime": [[det()]], "tensorrt": [[]]})

    assert main(["clip.mp4", "--rounds", "1", "--out", str(tmp_path / "c.json")]) == 1
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["agrees"] is False
    assert "FAIL" in capsys.readouterr().out


def test_main_passes_max_frames_on(tmp_path, fake_runs):
    fake = fake_runs(SAME)

    main(["clip.mp4", "--max-frames", "300", "--rounds", "1", "--out", str(tmp_path / "c.json")])

    assert {c[4] for c in fake.calls} == {300}


def test_main_fails_when_the_runtimes_timed_different_frame_counts(tmp_path, fake_runs):
    fake_runs(SAME, {"onnxruntime": timing_log(30.0, 40.0, n=5), "tensorrt": timing_log(5.0, 15.0, n=4)})

    with pytest.raises(SystemExit, match="5 frames .* 4"):
        main(["clip.mp4", "--out", str(tmp_path / "c.json")])


def test_run_pipeline_asks_tether_pipeline_for_the_runtime(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(runtime_compare, "cpp_binary", lambda cpp_dir, name: tmp_path / name)
    monkeypatch.setattr(runtime_compare, "_run", lambda command, name: commands.append([str(c) for c in command]))

    runtime_compare.run_pipeline(tmp_path, "clip.mp4", "m.onnx", "tensorrt", ["--dump", "d.jsonl"], None)
    runtime_compare.run_pipeline(tmp_path, "clip.mp4", "m.onnx", "onnxruntime", ["--timing", "t.jsonl"], 50)

    binary = str(tmp_path / "tether-pipeline")
    assert commands == [
        [binary, "clip.mp4", "--model", "m.onnx", "--runtime", "tensorrt", "--dump", "d.jsonl"],
        [binary, "clip.mp4", "--model", "m.onnx", "--runtime", "onnxruntime", "--timing", "t.jsonl",
         "--max-frames", "50"],
    ]


def test_a_missing_cpp_binary_says_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit, match="build it with cpp/build.sh"):
        runtime_compare.run_pipeline(tmp_path, "clip.mp4", "m.onnx", "tensorrt", [], None)
