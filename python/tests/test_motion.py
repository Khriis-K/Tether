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


def test_fast_objects_crossing_paths_do_not_swap_ids():
    # Same lane (4 px apart in y), 30 px/frame each way; they pass through
    # each other around frame 10, so mid-crossing both detections overlap
    # both predictions.
    def rightward(f):
        return box_at(30.0 * f, y=50.0)

    def leftward(f):
        return box_at(600.0 - 30.0 * f, y=54.0)

    tracker = Tracker()
    outputs = [tracker.update([Detection(box=rightward(f), score=0.9, cls=0),
                               Detection(box=leftward(f), score=0.9, cls=0)])
               for f in range(20)]

    ids_by_object = [({t.box: t.id for t in tracks}[rightward(f)],
                      {t.box: t.id for t in tracks}[leftward(f)])
                     for f, tracks in enumerate(outputs)]
    assert ids_by_object[0][0] != ids_by_object[0][1]
    assert ids_by_object == [ids_by_object[0]] * 20


def test_object_whose_box_shrinks_sharply_keeps_its_id():
    # 40 px box, then a 12 px box at the same center (e.g. the detector now
    # sees only part of it). One shrink is not a shrink rate: the predicted
    # box must stay a real box near the object.
    def centered(size):
        return (100.0 - size / 2, 100.0 - size / 2, 100.0 + size / 2, 100.0 + size / 2)

    tracker = Tracker()
    outputs = [tracker.update([Detection(box=centered(s), score=0.9, cls=0)])
               for s in [40.0] + [12.0] * 5]

    ids = single_ids(outputs)
    assert ids == [[ids[0][0]]] * 6
    for tracks in outputs:
        x1, y1, x2, y2 = tracks[0].predicted_box
        assert x2 > x1 and y2 > y1


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
