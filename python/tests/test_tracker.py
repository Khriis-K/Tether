from tether.formats import Detection
from tether.tracker import Tracker


def moving_box(frame, start_x=0.0, speed=2.0, y=50.0, size=20.0):
    x = start_x + speed * frame
    return (x, y, x + size, y + size)


def run(frames):
    tracker = Tracker()
    return [tracker.update(dets) for dets in frames]


def test_steadily_moving_object_keeps_the_same_id_every_frame():
    frames = [[Detection(box=moving_box(f), score=0.9, cls=0)] for f in range(50)]

    outputs = run(frames)

    ids = [[t.id for t in tracks] for tracks in outputs]
    assert ids == [[ids[0][0]]] * 50


def test_track_reports_box_class_status_and_frames_since_seen():
    tracks = run([[Detection(box=(10.0, 20.0, 30.0, 40.0), score=0.9, cls=3)]])[0]

    assert len(tracks) == 1
    track = tracks[0]
    assert track.box == (10.0, 20.0, 30.0, 40.0)
    assert track.cls == 3
    assert track.status == "tentative"  # new tracks are confirmed later (test_lifecycle.py)
    assert track.frames_since_seen == 0


def test_track_box_follows_the_matched_detection():
    frames = [[Detection(box=moving_box(f), score=0.9, cls=0)] for f in range(3)]

    outputs = run(frames)

    assert [tracks[0].box for tracks in outputs] == [moving_box(f) for f in range(3)]


def test_recent_path_holds_box_centers_oldest_first():
    frames = [[Detection(box=moving_box(f), score=0.9, cls=0)] for f in range(3)]

    track = run(frames)[-1][0]

    assert track.path == ((10.0, 60.0), (12.0, 60.0), (14.0, 60.0))


def test_recent_path_is_capped():
    frames = [[Detection(box=moving_box(f), score=0.9, cls=0)] for f in range(200)]

    track = run(frames)[-1][0]

    # Only frames 170..199 remain: centers x = 2f + 10.
    assert len(track.path) == 30
    assert track.path[0] == (350.0, 60.0)
    assert track.path[-1] == (408.0, 60.0)


def test_two_separate_objects_get_distinct_stable_ids():
    frames = [
        [
            Detection(box=moving_box(f, start_x=0.0), score=0.9, cls=0),
            Detection(box=moving_box(f, start_x=300.0), score=0.9, cls=0),
        ]
        for f in range(10)
    ]

    outputs = run(frames)

    first_ids = [t.id for t in outputs[0]]
    assert len(set(first_ids)) == 2
    for tracks in outputs:
        assert [t.id for t in tracks] == first_ids


def test_detections_of_different_classes_do_not_match():
    frames = [
        [Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=0)],
        [Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=1)],
    ]

    outputs = run(frames)

    assert outputs[0][0].id != outputs[1][0].id


def test_new_object_gets_a_new_id_that_is_never_reused():
    frames = [
        [Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=0)],
        [],
        [Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=0)],
    ]

    outputs = run(frames)

    assert outputs[1] == []
    assert outputs[2][0].id != outputs[0][0].id


def test_confirmed_track_gets_a_detection_before_a_tentative_track_that_overlaps_it_more():
    confirmed_box = (0.0, 0.0, 20.0, 20.0)
    tentative_box = (10.0, 0.0, 30.0, 20.0)
    shared = (8.0, 0.0, 28.0, 20.0)  # IoU 0.43 with the confirmed track, 0.82 with the tentative one
    frames = [
        [Detection(box=confirmed_box, score=0.9, cls=0)],
        [Detection(box=confirmed_box, score=0.9, cls=0)],
        [Detection(box=confirmed_box, score=0.9, cls=0),
         Detection(box=tentative_box, score=0.9, cls=0)],
        [Detection(box=shared, score=0.9, cls=0)],
    ]

    outputs = run(frames)

    confirmed = next(t for t in outputs[2] if t.box == confirmed_box)
    assert confirmed.status == "tracked"
    assert [(t.id, t.status, t.box) for t in outputs[3]] == [(confirmed.id, "tracked", shared)]


def test_detections_below_the_score_threshold_are_ignored():
    tracker = Tracker(min_score=0.5)

    tracks = tracker.update([
        Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.49, cls=0),
        Detection(box=(50.0, 50.0, 60.0, 60.0), score=0.5, cls=0),
    ])

    assert [t.box for t in tracks] == [(50.0, 50.0, 60.0, 60.0)]


def test_a_low_score_detection_does_not_keep_a_track_matched():
    tracker = Tracker(min_score=0.5, confirm_frames=1)
    first = tracker.update([Detection(box=moving_box(0), score=0.9, cls=0)])

    tracks = tracker.update([Detection(box=moving_box(1), score=0.2, cls=0)])

    assert [(t.id, t.status) for t in tracks] == [(first[0].id, "searching")]
