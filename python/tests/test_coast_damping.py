"""Velocity damping while a track coasts.

Each frame a track goes unmatched, its velocity is multiplied by coast_damping
before the prediction steps forward, so a coasted prediction slows down instead
of running off at the last velocity. 1 is undamped; 0 holds the position still.
Scripted object moving right at SPEED px/frame.
"""

import pytest

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import Tracker

SIZE = 20.0
SPEED = 10.0
CONFIRM = 3
MATCHED, MISSED = 10, 5


def det_at(frame):
    x = SPEED * frame
    return Detection(box=(x, 0.0, x + SIZE, SIZE), score=0.9, cls=0)


def moving_then_missing(matched=MATCHED, missed=MISSED):
    return [[det_at(f)] for f in range(matched)] + [[] for _ in range(missed)]


def run(frames, coast_damping, **kwargs):
    tracker = Tracker(confirm_frames=CONFIRM, coast_damping=coast_damping, **kwargs)
    return [tracker.update(dets) for dets in frames]


def only(tracks):
    assert len(tracks) == 1
    return tracks[0]


def predicted_x(outputs):
    return [only(tracks).predicted_box[0] for tracks in outputs]


def steps(xs):
    return [b - a for a, b in zip(xs, xs[1:])]


@pytest.mark.parametrize("damping", [0.5, 0.8])
def test_velocity_shrinks_by_the_damping_each_coasted_frame(damping):
    outputs = run(moving_then_missing(), damping)

    # predicted_x[MATCHED - 1] was predicted on the last matched frame; every
    # later one is a coasted step.
    coasted = steps(predicted_x(outputs)[MATCHED - 1:])
    assert len(coasted) == MISSED
    assert coasted[0] > 0
    for before, after in zip(coasted, coasted[1:]):
        assert after == pytest.approx(damping * before)


def test_first_coasted_step_is_damped_too():
    undamped = steps(predicted_x(run(moving_then_missing(), 1.0))[MATCHED - 1:])
    damped = steps(predicted_x(run(moving_then_missing(), 0.5))[MATCHED - 1:])

    assert damped[0] == pytest.approx(0.5 * undamped[0])


def test_zero_damping_holds_the_coasted_position_still():
    xs = predicted_x(run(moving_then_missing(), 0.0))

    assert xs[MATCHED - 1:] == [xs[MATCHED - 1]] * (MISSED + 1)


def test_matched_frames_are_not_damped():
    frames = moving_then_missing(missed=0)

    assert run(frames, 0.5) == run(frames, 1.0)


def test_velocity_recovers_once_the_track_matches_again():
    # Missed for a few frames, then matched again (by the search gate, as the
    # damped prediction lags the object) for many frames at the same speed.
    gap = 4
    frames = ([[det_at(f)] for f in range(MATCHED)] + [[] for _ in range(gap)]
              + [[det_at(f)] for f in range(MATCHED + gap, MATCHED + gap + 30)])

    outputs = run(frames, 0.5, search_gate=5.0)

    assert len({only(tracks).id for tracks in outputs}) == 1
    # Damping that stuck would leave the prediction lagging behind the object;
    # undamped, it lands where the object is on the next frame.
    assert predicted_x(outputs)[-1] == pytest.approx(SPEED * len(frames), abs=0.5)


def test_default_is_undamped():
    frames = moving_then_missing()
    tracker = Tracker(confirm_frames=CONFIRM)

    assert [tracker.update(dets) for dets in frames] == run(frames, 1.0)


def test_cli_passes_the_damping_to_the_tracker(tmp_path):
    save_detections(tmp_path / "dets.jsonl", moving_then_missing())

    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "tracks.jsonl"),
          "--confirm-frames", str(CONFIRM), "--coast-damping", "0"])

    xs = predicted_x(load_tracks(tmp_path / "tracks.jsonl"))
    assert xs[MATCHED - 1:] == [xs[MATCHED - 1]] * (MISSED + 1)
