import json
import random

import pytest

from tether.formats import STEPS, load_timings, save_timings
from tether.timing import main, p99, summarize

# Three frames run one after another, steps back to back (the sequential pipeline).
SEQUENTIAL = [
    {"capture": (0.0, 2.0), "prep": (2.0, 3.0), "detect": (3.0, 13.0), "track": (13.0, 14.0), "draw": (14.0, 16.0)},
    {"capture": (16.0, 20.0), "prep": (20.0, 21.0), "detect": (21.0, 31.0), "track": (31.0, 32.0), "draw": (32.0, 34.0)},
    {"capture": (34.0, 36.0), "prep": (36.0, 37.0), "detect": (37.0, 57.0), "track": (57.0, 58.0), "draw": (58.0, 60.0)},
]

# The same kind of frames overlapping, as a parallel pipeline runs them: the next
# frame is captured while this one is detected, then waits for the detector.
PARALLEL = [
    {"capture": (0.0, 5.0), "prep": (5.0, 6.0), "detect": (6.0, 16.0), "track": (16.0, 17.0), "draw": (17.0, 19.0)},
    {"capture": (5.0, 10.0), "prep": (10.0, 11.0), "detect": (16.0, 26.0), "track": (26.0, 27.0), "draw": (27.0, 29.0)},
    {"capture": (10.0, 15.0), "prep": (15.0, 16.0), "detect": (26.0, 36.0), "track": (36.0, 37.0), "draw": (37.0, 39.0)},
]


def test_p99_of_fewer_than_100_frames_is_the_worst_frame():
    assert p99([3.0, 9.0, 1.0, 2.0]) == 9.0


@pytest.mark.parametrize("n, expected", [
    (100, 99.0),   # the single worst frame is the worst 1%, so it's left out
    (101, 100.0),  # rank ceil(99.99) = 100
    (200, 198.0),  # two worst frames left out
    (1000, 990.0),
])
def test_p99_is_the_nearest_rank_value(n, expected):
    values = [float(v) for v in range(1, n + 1)]
    random.Random(n).shuffle(values)

    assert p99(values) == expected


def test_p99_of_a_single_value_is_that_value():
    assert p99([4.5]) == 4.5


def test_summary_of_a_sequential_run():
    summary = summarize(SEQUENTIAL)

    assert summary["frames"] == 3
    # Frames finish at 16, 34 and 60: intervals 18 and 26.
    assert summary["frame_time_ms"] == {"mean": pytest.approx(22.0), "p99": 26.0}
    assert summary["fps"] == pytest.approx(1000 / 22)
    # Frame in hand (capture end) to on screen (draw end): 14, 14, 24.
    assert summary["end_to_end_delay_ms"] == {"mean": pytest.approx(52 / 3), "p99": 24.0}
    assert summary["steps_ms"] == {
        "capture": {"mean": pytest.approx(8 / 3), "p99": 4.0},
        "prep": {"mean": pytest.approx(1.0), "p99": 1.0},
        "detect": {"mean": pytest.approx(40 / 3), "p99": 20.0},
        "track": {"mean": pytest.approx(1.0), "p99": 1.0},
        "draw": {"mean": pytest.approx(2.0), "p99": 2.0},
    }


def test_frame_time_and_delay_come_apart_when_frames_overlap():
    summary = summarize(PARALLEL)

    # Each frame's steps add up to 19 ms, but a frame finishes every 10 ms.
    assert summary["frame_time_ms"] == {"mean": pytest.approx(10.0), "p99": 10.0}
    # Waiting for the detector counts toward delay: 14, 19, 24.
    assert summary["end_to_end_delay_ms"] == {"mean": pytest.approx(19.0), "p99": 24.0}
    assert summary["steps_ms"]["detect"] == {"mean": pytest.approx(10.0), "p99": 10.0}


def test_a_perfectly_steady_run_has_p99_equal_to_the_mean():
    frames = [{step: (10.0 * i + j, 10.0 * i + j + 2.0) for j, step in enumerate(STEPS)} for i in range(5)]

    summary = summarize(frames)

    assert summary["frame_time_ms"]["p99"] == pytest.approx(summary["frame_time_ms"]["mean"])
    assert summary["end_to_end_delay_ms"]["p99"] == pytest.approx(summary["end_to_end_delay_ms"]["mean"])


def test_summary_needs_two_frames_to_measure_a_frame_time():
    with pytest.raises(ValueError, match="at least 2 frames"):
        summarize(SEQUENTIAL[:1])


def test_timing_log_round_trips(tmp_path):
    path = tmp_path / "timing.jsonl"

    save_timings(path, SEQUENTIAL)

    assert load_timings(path) == SEQUENTIAL
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first == {"frame": 0, "steps": {"capture": [0.0, 2.0], "prep": [2.0, 3.0], "detect": [3.0, 13.0],
                                           "track": [13.0, 14.0], "draw": [14.0, 16.0]}}


def test_timing_log_without_every_step_is_rejected(tmp_path):
    path = tmp_path / "timing.jsonl"
    path.write_text('{"frame": 0, "steps": {"capture": [0.0, 1.0]}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="frame 0 is missing steps"):
        load_timings(path)


def test_tool_saves_the_summary_next_to_the_log(tmp_path, capsys):
    log = tmp_path / "timing-run1.jsonl"
    save_timings(log, SEQUENTIAL)

    assert main([str(log)]) == 0

    saved = json.loads((tmp_path / "timing-run1.summary.json").read_text(encoding="utf-8"))
    assert saved == summarize(SEQUENTIAL)
    assert "P99" in capsys.readouterr().out


def test_tool_saves_the_summary_where_asked(tmp_path):
    log = tmp_path / "timing.jsonl"
    out = tmp_path / "results" / "summary.json"
    save_timings(log, SEQUENTIAL)

    main([str(log), "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8")) == summarize(SEQUENTIAL)
