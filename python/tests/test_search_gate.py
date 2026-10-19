"""Re-matching a searching track by center distance, with a gate that grows while it searches.

A searching track's coasted box can drift off its object, so IoU can no longer
pair them. The gate is search_gate + search_gate_growth * frames_since_seen, in
track box sizes, where frames_since_seen is what the track reported on the frame
before. Scripted, stationary objects: the prediction stays where the box was.
"""

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.matching import MIN_IOU, iou, match_by_distance
from tether.tracker import Tracker

SIZE = 20.0
X, Y = 100.0, 100.0
CONFIRM = 3
SEARCH_GATE, SEARCH_GATE_GROWTH = 1.0, 0.5  # test values; the tracker's default is off


def box_at(x, y=Y, size=SIZE):
    return (x, y, x + size, y + size)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def gate(frames_since_seen):
    return SEARCH_GATE + SEARCH_GATE_GROWTH * frames_since_seen


def confirmed_then_missing(missed):
    """A stationary object confirmed over a few frames, then undetected for `missed` frames."""
    return [[det(box_at(X))] for _ in range(CONFIRM + 2)] + [[] for _ in range(missed)]


def run(frames, search_gate=SEARCH_GATE, search_gate_growth=SEARCH_GATE_GROWTH):
    tracker = Tracker(confirm_frames=CONFIRM, search_gate=search_gate,
                      search_gate_growth=search_gate_growth)
    return [tracker.update(dets) for dets in frames]


def only(tracks):
    assert len(tracks) == 1
    return tracks[0]


def by_box(tracks, box):
    matches = [t for t in tracks if t.box == box and t.frames_since_seen == 0]
    assert len(matches) == 1
    return matches[0]


def test_searching_track_rematches_a_detection_outside_iou_range_but_inside_the_gate():
    offset = max(1.25, gate(1) - 0.25)  # box sizes: at least 1.25 so the boxes don't overlap
    assert offset < gate(1)
    returned = box_at(X + offset * SIZE)
    assert iou(box_at(X), returned) < MIN_IOU

    outputs = run(confirmed_then_missing(1) + [[det(returned)]])

    searching = only(outputs[-2])
    assert searching.status == "searching" and searching.frames_since_seen == 1
    track = only(outputs[-1])
    assert track.id == searching.id
    assert track.status == "tracked"
    assert track.box == returned


def test_gate_widens_with_frames_since_seen():
    # Outside the gate after one missed frame, inside it after five.
    offset = (gate(1) + gate(5)) / 2
    assert gate(1) < offset < gate(5)
    returned = box_at(X + offset * SIZE)

    after_one = run(confirmed_then_missing(1) + [[det(returned)]])
    after_five = run(confirmed_then_missing(5) + [[det(returned)]])

    old_id = only(after_one[-2]).id
    assert by_box(after_one[-1], returned).id != old_id
    assert only(after_five[-1]).id == only(after_five[-2]).id
    assert only(after_five[-1]).status == "tracked"


def test_detection_outside_the_gate_starts_a_new_track():
    returned = box_at(X + (gate(1) + 0.25) * SIZE)

    outputs = run(confirmed_then_missing(1) + [[det(returned)]])

    old = only(outputs[-2])
    tracks = {t.id: t for t in outputs[-1]}
    assert tracks[old.id].status == "searching"
    new = by_box(outputs[-1], returned)
    assert new.id != old.id
    assert new.status == "tentative"


def test_searching_track_takes_its_detection_back_from_the_tentative_track_it_started():
    # The failure on Anti-UAV: a track misses one jump, the detection starts a
    # tentative track there, and on the next frame that tentative track overlaps
    # the detection fully. The searching track should still get it.
    offset = max(1.25, min(gate(0), gate(1)) - 0.25)
    assert offset < gate(1)
    jumped = box_at(X + offset * SIZE)
    assert iou(box_at(X), jumped) < MIN_IOU

    frames = [[det(box_at(X))] for _ in range(CONFIRM + 2)] + [[det(jumped)], [det(jumped)]]
    outputs = run(frames)

    original = only(outputs[CONFIRM + 1]).id
    assert by_box(outputs[-1], jumped).id == original
    assert by_box(outputs[-1], jumped).status == "tracked"


def returning_outside_iou_range():
    """A searching track and a detection inside its gate after one missed frame, but not overlapping."""
    returned = box_at(X + 1.25 * SIZE)
    assert iou(box_at(X), returned) < MIN_IOU and 1.25 < gate(1)
    return confirmed_then_missing(1) + [[det(returned)]], returned


# A much smaller box with the same center: IoU below MIN_IOU, but distance 0, so any gate above 0 takes it.
CENTERED = (X + 8.0, Y + 8.0, X + 12.0, Y + 12.0)


def test_gate_is_off_by_default():
    assert iou(box_at(X), CENTERED) < MIN_IOU
    tracker = Tracker(confirm_frames=CONFIRM)

    outputs = [tracker.update(dets) for dets in confirmed_then_missing(1) + [[det(CENTERED)]]]

    assert by_box(outputs[-1], CENTERED).id != only(outputs[-2]).id


def test_zero_gate_never_matches_even_a_detection_centered_on_the_prediction():
    assert iou(box_at(X), CENTERED) < MIN_IOU

    outputs = run(confirmed_then_missing(1) + [[det(CENTERED)]], search_gate=0.0, search_gate_growth=0.0)

    assert by_box(outputs[-1], CENTERED).id != only(outputs[-2]).id


def test_cli_passes_the_gate_settings_to_the_tracker(tmp_path):
    frames, returned = returning_outside_iou_range()
    save_detections(tmp_path / "dets.jsonl", frames)

    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "off.jsonl"), "--confirm-frames", str(CONFIRM)])
    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "on.jsonl"), "--confirm-frames", str(CONFIRM),
          "--search-gate", str(SEARCH_GATE), "--search-gate-growth", str(SEARCH_GATE_GROWTH)])

    off, on = load_tracks(tmp_path / "off.jsonl"), load_tracks(tmp_path / "on.jsonl")
    assert by_box(off[-1], returned).id != only(off[-2]).id
    assert only(on[-1]).id == only(on[-2]).id


def test_match_by_distance_applies_each_track_its_own_gate():
    tracks = [box_at(0.0), box_at(1000.0)]
    # Detection 0 is 2 box sizes from track 0; detection 1 is 2 box sizes from track 1.
    dets = [box_at(2 * SIZE), box_at(1000.0 + 2 * SIZE), box_at(5000.0)]

    assert match_by_distance(tracks, dets, max_distance=[3.0, 1.0]) == [(0, 0)]
    assert match_by_distance(tracks, dets, max_distance=[1.0, 3.0]) == [(1, 1)]
