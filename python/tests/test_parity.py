import dataclasses
import json
import subprocess
import time

import pytest

from tether import parity
from tether.formats import Detection, Track, save_detections, save_tracks
from tether.parity import first_mismatch, main, max_box_difference, scripted_inputs, time_python
from tether.tracker import Tracker


def track(id=1, status="tracked", box=(0.0, 0.0, 10.0, 10.0), **changes):
    t = Track(id=id, box=box, cls=0, status=status, predicted_box=box, frames_since_seen=0,
              path=((5.0, 5.0),))
    return dataclasses.replace(t, **changes)


def shifted(t, dx, field="box"):
    x1, y1, x2, y2 = getattr(t, field)
    return dataclasses.replace(t, **{field: (x1 + dx, y1, x2 + dx, y2)})


FRAMES = [[track(1)], [track(1), track(2)], [track(1), track(2, status="searching")]]


def test_identical_outputs_match():
    assert first_mismatch(FRAMES, FRAMES) is None


def test_different_id_is_reported_at_its_frame_and_track():
    other = [FRAMES[0], [track(1), track(3)], FRAMES[2]]

    m = first_mismatch(FRAMES, other)

    assert (m.frame, m.track_id) == (1, 2)
    assert "3" in m.what


def test_different_status_is_reported_at_its_frame_and_track():
    other = [FRAMES[0], FRAMES[1], [track(1), track(2, status="tracked")]]

    m = first_mismatch(FRAMES, other)

    assert (m.frame, m.track_id) == (2, 2)
    assert "searching" in m.what and "tracked" in m.what


def test_only_the_first_differing_frame_is_reported():
    other = [FRAMES[0], [track(1, status="tentative"), track(2)], [track(1), track(2)]]

    assert first_mismatch(FRAMES, other).frame == 1


@pytest.mark.parametrize("field", ["box", "predicted_box"])
def test_boxes_within_the_tolerance_match(field):
    other = [[shifted(t, 1e-7, field) for t in f] for f in FRAMES]

    assert first_mismatch(FRAMES, other, tolerance=1e-6) is None


@pytest.mark.parametrize("field", ["box", "predicted_box"])
def test_a_box_beyond_the_tolerance_is_reported(field):
    other = [FRAMES[0], [track(1), shifted(track(2), 1e-3, field)], FRAMES[2]]

    m = first_mismatch(FRAMES, other, tolerance=1e-6)

    assert (m.frame, m.track_id) == (1, 2)
    assert field in m.what


@pytest.mark.parametrize("field", ["box", "predicted_box"])
def test_a_nan_box_is_a_mismatch(field):
    # abs(nan - x) > tolerance is False, so a NaN must not slip through as "within tolerance".
    nan = float("nan")
    other = [[dataclasses.replace(track(1), **{field: (nan, 0.0, 10.0, 10.0)})], FRAMES[1], FRAMES[2]]

    assert first_mismatch(FRAMES, other).frame == 0


def test_a_nan_path_point_is_a_mismatch():
    other = [[track(1, path=((float("nan"), 5.0),))], FRAMES[1], FRAMES[2]]

    assert first_mismatch(FRAMES, other).frame == 0


def test_a_path_point_beyond_the_tolerance_is_reported():
    other = [[track(1, path=((5.0, 5.1),))], FRAMES[1], FRAMES[2]]

    assert (first_mismatch(FRAMES, other).frame, first_mismatch(FRAMES, other).track_id) == (0, 1)


@pytest.mark.parametrize("changes", [{"cls": 2}, {"frames_since_seen": 1},
                                     {"path": ((5.0, 5.0), (6.0, 6.0))}])
def test_class_missed_frames_and_path_length_must_match_exactly(changes):
    other = [[track(1, **changes)], FRAMES[1], FRAMES[2]]

    assert first_mismatch(FRAMES, other).frame == 0


def test_a_track_only_one_side_has_is_reported():
    other = [FRAMES[0], [track(1)], FRAMES[2]]

    m = first_mismatch(FRAMES, other)
    assert (m.frame, m.track_id) == (1, 2)
    assert (first_mismatch(other, FRAMES).frame, first_mismatch(other, FRAMES).track_id) == (1, 2)


def test_a_different_frame_count_is_reported_where_one_side_ends():
    m = first_mismatch(FRAMES, FRAMES[:2])

    assert m.frame == 2
    assert m.track_id is None


def test_max_box_difference_covers_boxes_and_predicted_boxes():
    other = [FRAMES[0], [track(1), shifted(track(2), 0.5, "predicted_box")], FRAMES[2]]

    assert max_box_difference(FRAMES, FRAMES) == 0.0
    assert max_box_difference(FRAMES, other) == pytest.approx(0.5)


def test_scripted_inputs_are_fixed_and_exercise_the_tracker():
    inputs = scripted_inputs()

    assert len(inputs) >= 5
    assert inputs == scripted_inputs()
    for name, frames in inputs.items():
        tracker = Tracker()
        outputs = [tracker.update(d) for d in frames]
        assert any(t.status == "tracked" for f in outputs for t in f), name


def test_scripted_inputs_cover_searching_and_removed_tracks():
    statuses, ids = set(), set()
    for frames in scripted_inputs().values():
        tracker = Tracker()
        for dets in frames:
            out = tracker.update(dets)
            statuses |= {t.status for t in out}
            ids |= {t.id for t in out}

    assert statuses == {"tentative", "tracked", "searching"}
    assert len(ids) > 5


def test_time_python_gives_one_time_per_repeat():
    frames = [[Detection(box=(5.0 * f, 0.0, 5.0 * f + 20, 20.0), score=0.9, cls=0)] for f in range(500)]

    times = time_python(frames, repeats=3)

    assert len(times) == 3
    assert all(ms > 0 for ms in times)


def test_time_python_times_the_tracker_updates(monkeypatch):
    def slow_update(self, detections):
        time.sleep(0.002)
        return []
    monkeypatch.setattr(parity.Tracker, "update", slow_update)

    times = time_python([[]] * 20, repeats=2)

    assert all(ms >= 20 * 2.0 for ms in times)


# main() with stand-ins for the C++ binaries, so these run without a C++ build.

class FakeCpp:
    """Runs the Python tracker in place of tether-track, optionally breaking one track's status."""

    def __init__(self, break_frame=None, cpp_ms=1.0):
        self.break_frame = break_frame
        self.cpp_ms = cpp_ms

    def track(self, cpp_dir, detections_path, tracks_path):
        from tether.formats import load_detections
        tracker = Tracker()
        frames = [tracker.update(d) for d in load_detections(detections_path)]
        if self.break_frame is not None:
            t = frames[self.break_frame][0]
            frames[self.break_frame][0] = dataclasses.replace(t, status="searching")
        save_tracks(tracks_path, frames)

    def time(self, cpp_dir, detections_path, repeats):
        return [self.cpp_ms] * repeats


@pytest.fixture
def fake_cpp(monkeypatch):
    def install(fake):
        monkeypatch.setattr(parity, "run_cpp_tracker", fake.track)
        monkeypatch.setattr(parity, "time_cpp", fake.time)
        monkeypatch.setattr(parity, "time_python", lambda frames, repeats: [4.0] * repeats)
    return install


def walk_file(tmp_path, n=10):
    path = tmp_path / "walk.jsonl"
    save_detections(path, [[Detection(box=(5.0 * f, 0.0, 5.0 * f + 20, 20.0), score=0.9, cls=0)]
                           for f in range(n)])
    return path


def test_main_passes_and_saves_the_speedup_when_outputs_match(tmp_path, fake_cpp, capsys):
    fake_cpp(FakeCpp(cpp_ms=1.0))
    out = tmp_path / "parity.json"

    code = main([str(walk_file(tmp_path)), "--scripted", "--repeats", "2", "--out", str(out)])

    assert code == 0
    result = json.loads(out.read_text())
    assert result["all_match"] is True
    names = [r["name"] for r in result["inputs"]]
    assert names[0] == "walk.jsonl"
    assert set(scripted_inputs()) <= set(names)
    first = result["inputs"][0]
    assert first["frames"] == 10
    assert first["match"] is True and first["first_mismatch"] is None
    assert first["python_ms"] == 4.0 and first["cpp_ms"] == 1.0
    assert first["speedup"] == 4.0
    assert result["speedup"] == 4.0
    assert "4.0x" in capsys.readouterr().out


def test_main_fails_and_points_at_the_first_differing_frame(tmp_path, fake_cpp, capsys):
    fake_cpp(FakeCpp(break_frame=6))
    out = tmp_path / "parity.json"

    code = main([str(walk_file(tmp_path)), "--out", str(out)])

    assert code == 1
    result = json.loads(out.read_text())
    assert result["all_match"] is False
    assert result["inputs"][0]["first_mismatch"]["frame"] == 6
    assert result["inputs"][0]["first_mismatch"]["track_id"] == 1
    printed = capsys.readouterr().out
    assert "frame 6" in printed and "track 1" in printed


def test_main_keeps_the_median_of_the_repeats(tmp_path, fake_cpp, monkeypatch):
    fake_cpp(FakeCpp())
    monkeypatch.setattr(parity, "time_python", lambda frames, repeats: [1.0, 9.0, 4.0])
    monkeypatch.setattr(parity, "time_cpp", lambda cpp_dir, path, repeats: [2.0, 1.0, 8.0])
    out = tmp_path / "parity.json"

    main([str(walk_file(tmp_path)), "--repeats", "3", "--out", str(out)])

    first = json.loads(out.read_text())["inputs"][0]
    assert (first["python_ms"], first["cpp_ms"]) == (4.0, 2.0)


def test_a_failing_cpp_binary_stops_with_its_error(tmp_path, monkeypatch):
    (tmp_path / "tether-track").write_text("")  # found, but "runs" via the stand-in below
    monkeypatch.setattr(parity.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "bad detection file"))

    with pytest.raises(SystemExit) as e:
        parity.run_cpp_tracker(tmp_path, tmp_path / "d.jsonl", tmp_path / "t.jsonl")
    assert "bad detection file" in str(e.value.code)


def test_a_missing_cpp_binary_says_how_to_build_it(tmp_path):
    with pytest.raises(SystemExit) as e:
        parity.time_cpp(tmp_path, tmp_path / "d.jsonl", 1)
    assert "cpp/build.sh" in str(e.value.code)


def test_main_needs_an_input(tmp_path, fake_cpp):
    fake_cpp(FakeCpp())

    with pytest.raises(SystemExit) as e:
        main(["--out", str(tmp_path / "p.json")])
    assert e.value.code == 2
