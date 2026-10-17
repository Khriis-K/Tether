"""Motion behavior: scripted detections in, tracks out. No filter internals."""

import pytest

from tether.formats import Detection
from tether.tracker import Tracker

SIZE = 20.0


def box_at(x, y=50.0, size=SIZE):
    return (x, y, x + size, y + size)


def run(xs):
    """One detection per frame, its box's left edge at each x."""
    tracker = Tracker()
    return [tracker.update([Detection(box=box_at(x), score=0.9, cls=0)]) for x in xs]


def single_ids(outputs):
    return [[t.id for t in tracks] for tracks in outputs]


def test_object_jumping_far_every_frame_keeps_the_same_id():
    # Speeds up smoothly from 2 to 60 px/frame, then holds 60. Once the step
    # passes the box size (20) the boxes stop overlapping between frames, so
    # only the predicted box can match.
    xs, x, speed = [], 0.0, 2.0
    for _ in range(50):
        xs.append(x)
        x += speed
        speed = min(speed + 4.0, 60.0)
    assert xs[-1] - xs[-2] == 60.0

    outputs = run(xs)

    ids = single_ids(outputs)
    assert ids == [[ids[0][0]]] * 50


def test_object_already_fast_on_its_first_frame_keeps_the_same_id():
    # 50 px/frame from the very first frame: no overlap, no velocity history.
    outputs = run([50.0 * f for f in range(20)])

    ids = single_ids(outputs)
    assert ids == [[ids[0][0]]] * 20


def test_fast_objects_moving_toward_each_other_keep_their_ids():
    tracker = Tracker()
    outputs = []
    for f in range(12):
        outputs.append(tracker.update([
            Detection(box=box_at(30.0 * f, y=0.0), score=0.9, cls=0),
            Detection(box=box_at(700.0 - 30.0 * f, y=200.0), score=0.9, cls=0),
        ]))

    by_lane = [{t.box[1]: t.id for t in tracks} for tracks in outputs]
    assert len(set(by_lane[0].values())) == 2
    assert by_lane == [by_lane[0]] * 12


def test_far_away_detection_after_a_new_track_gets_a_new_id():
    outputs = run([0.0, 500.0])

    assert outputs[1][0].id != outputs[0][0].id


def test_established_track_does_not_take_a_nearby_non_overlapping_detection():
    # A still object seen for 5 frames, then the only detection is 3 box sizes
    # away. The track's prediction says "still here", so that's someone else.
    outputs = run([0.0] * 5 + [60.0])

    assert outputs[5][0].id != outputs[0][0].id


def test_new_track_predicts_it_stays_where_it_is():
    track = run([10.0])[0][0]

    assert track.predicted_box == track.box


def test_predicted_box_leads_a_steadily_moving_object_to_its_next_box():
    track = run([3.0 * f for f in range(20)])[-1][0]

    assert track.predicted_box == pytest.approx(box_at(3.0 * 20), abs=0.5)
