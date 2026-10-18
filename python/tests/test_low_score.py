"""Low-score pass: detections scored between LOW_SCORE and min_score only keep tracked tracks alive."""

from tether.formats import Detection
from tether.tracker import Tracker

BOX = (0.0, 50.0, 20.0, 70.0)  # a still object, so every prediction is this box
SHIFTED = (8.0, 50.0, 28.0, 70.0)  # IoU 0.43 with BOX: matches by IoU, but not as a low detection
HIGH = 0.9
LOW = 0.15


def det(box, score):
    return Detection(box=box, score=score, cls=0)


def confirmed_tracker():
    tracker = Tracker(confirm_frames=3)
    for _ in range(3):
        tracks = tracker.update([det(BOX, HIGH)])
    assert [t.status for t in tracks] == ["tracked"]
    return tracker, tracks[0].id


def test_low_detection_keeps_a_tracked_track_tracked():
    tracker, track_id = confirmed_tracker()

    tracks = tracker.update([det(BOX, LOW)])

    assert [(t.id, t.status, t.frames_since_seen) for t in tracks] == [(track_id, "tracked", 0)]


def test_low_detection_below_the_iou_gate_does_not_keep_a_track_tracked():
    tracker, track_id = confirmed_tracker()

    tracks = tracker.update([det(SHIFTED, LOW)])

    assert [(t.id, t.status) for t in tracks] == [(track_id, "searching")]


def test_high_detection_at_the_same_overlap_still_matches():
    tracker, track_id = confirmed_tracker()

    tracks = tracker.update([det(SHIFTED, HIGH)])

    assert [(t.id, t.status) for t in tracks] == [(track_id, "tracked")]


def test_detection_at_or_below_the_low_floor_is_ignored():
    tracker, track_id = confirmed_tracker()

    tracks = tracker.update([det(BOX, 0.1)])

    assert [(t.id, t.status) for t in tracks] == [(track_id, "searching")]


def test_low_detection_never_starts_a_track():
    tracker = Tracker(confirm_frames=1)

    outputs = [tracker.update([det(BOX, LOW)]) for _ in range(5)]

    assert outputs == [[]] * 5


def test_low_detection_does_not_match_a_searching_track():
    tracker, track_id = confirmed_tracker()
    tracker.update([])  # missed: now searching

    tracks = tracker.update([det(BOX, LOW)])

    assert [(t.id, t.status, t.frames_since_seen) for t in tracks] == [(track_id, "searching", 2)]


def test_low_detection_does_not_match_a_tentative_track():
    tracker = Tracker(confirm_frames=3)
    tracker.update([det(BOX, HIGH)])  # tentative

    tracks = tracker.update([det(BOX, LOW)])

    assert tracks == []  # the tentative track missed, so it was dropped
