"""Capping a searching track's gate by the distance to the nearest other confirmed track.

The cap is search_gate_cap times the center distance from the track's prediction
to the nearest other confirmed track's prediction, in this track's box sizes,
and applies on top of the growing gate. Tentative tracks don't count: the
tentative track a missed detection starts is exactly what the gate takes back.
Scripted, stationary objects: the prediction stays where the box was.
"""

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.matching import MIN_IOU, iou
from tether.tracker import Tracker

SIZE = 20.0
X, Y = 200.0, 100.0
CONFIRM = 3
SEARCH_GATE, SEARCH_GATE_GROWTH = 1.0, 0.5  # gate after one missed frame: 1.5 box sizes
CAP = 0.5
RETURN_OFFSET = 1.25  # box sizes: inside the full gate, and no overlap with the old box


def box_at(x, y=Y, size=SIZE):
    return (x, y, x + size, y + size)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def run(frames, search_gate_cap=CAP):
    tracker = Tracker(confirm_frames=CONFIRM, search_gate=SEARCH_GATE,
                      search_gate_growth=SEARCH_GATE_GROWTH, search_gate_cap=search_gate_cap)
    return [tracker.update(dets) for dets in frames]


def searching_next_to(neighbor, returned):
    """Object A at X, confirmed then missed for one frame and back at `returned`.

    `neighbor` is the box of an object B detected on every frame, or None for A alone.
    """
    others = [] if neighbor is None else [det(neighbor)]
    frames = [[det(box_at(X))] + others for _ in range(CONFIRM + 2)]
    return frames + [others, [det(returned)] + others]


def a_kept_its_id(outputs, returned):
    searching = [t for t in outputs[-2] if t.status == "searching"]
    assert len(searching) == 1
    at_returned = [t for t in outputs[-1] if t.box == returned and t.frames_since_seen == 0]
    assert len(at_returned) == 1
    return at_returned[0].id == searching[0].id


def returned_at(offset):
    returned = box_at(X + offset * SIZE)
    if offset >= 1.0:
        assert iou(box_at(X), returned) < MIN_IOU
    return returned


def test_isolated_searching_track_rematches_within_its_full_gate():
    returned = returned_at(RETURN_OFFSET)

    assert a_kept_its_id(run(searching_next_to(None, returned)), returned)


def test_nearby_track_shrinks_the_gate_to_the_cap():
    # B is 2 box sizes away on the other side, so the cap is 1.0: the detection
    # at 1.25 is beyond it, though inside the full gate of 1.5.
    neighbor = box_at(X - 2.0 * SIZE)
    returned = returned_at(RETURN_OFFSET)
    assert CAP * 2.0 < RETURN_OFFSET < SEARCH_GATE + SEARCH_GATE_GROWTH

    outputs = run(searching_next_to(neighbor, returned))

    assert not a_kept_its_id(outputs, returned)
    new = [t for t in outputs[-1] if t.box == returned]
    assert new[0].status == "tentative"


def test_without_the_cap_the_same_detection_rematches():
    neighbor = box_at(X - 2.0 * SIZE)
    returned = returned_at(RETURN_OFFSET)

    assert a_kept_its_id(run(searching_next_to(neighbor, returned), search_gate_cap=float("inf")), returned)


def test_detection_inside_the_cap_still_rematches():
    neighbor = box_at(X - 2.0 * SIZE)
    returned = returned_at(0.75)  # under the cap of 1.0, and too little overlap for IoU
    assert iou(box_at(X), returned) < MIN_IOU

    assert a_kept_its_id(run(searching_next_to(neighbor, returned)), returned)


def test_cap_measures_to_other_tracks_never_the_track_itself():
    # B is 3 box sizes away, so the cap is 1.5 and doesn't bind. Measured to
    # itself the distance would be 0, and the track could never re-match.
    neighbor = box_at(X - 3.0 * SIZE)
    returned = returned_at(RETURN_OFFSET)

    assert a_kept_its_id(run(searching_next_to(neighbor, returned)), returned)


def test_tentative_tracks_dont_cap_the_gate():
    # A misses a jump, the detection starts a tentative track there, and on the
    # next frame A should still take its detection back from it.
    jumped = returned_at(RETURN_OFFSET)
    frames = [[det(box_at(X))] for _ in range(CONFIRM + 2)] + [[det(jumped)], [det(jumped)]]

    outputs = run(frames)

    original = outputs[CONFIRM + 1][0].id
    at_jumped = [t for t in outputs[-1] if t.box == jumped]
    assert len(at_jumped) == 1
    assert at_jumped[0].id == original and at_jumped[0].status == "tracked"


def test_cap_is_on_by_default():
    # The default gate after one missed frame (1.75) reaches the detection at
    # 1.25; the default cap (half the 2 box sizes to B) doesn't.
    neighbor = box_at(X - 2.0 * SIZE)
    returned = returned_at(RETURN_OFFSET)
    tracker = Tracker()

    outputs = [tracker.update(dets) for dets in searching_next_to(neighbor, returned)]

    assert not a_kept_its_id(outputs, returned)


def test_cli_passes_the_cap_to_the_tracker(tmp_path):
    neighbor = box_at(X - 2.0 * SIZE)
    returned = returned_at(RETURN_OFFSET)
    save_detections(tmp_path / "dets.jsonl", searching_next_to(neighbor, returned))
    gate = ["--confirm-frames", str(CONFIRM), "--search-gate", str(SEARCH_GATE),
            "--search-gate-growth", str(SEARCH_GATE_GROWTH)]

    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "off.jsonl"), *gate, "--search-gate-cap", "inf"])
    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "on.jsonl"), *gate, "--search-gate-cap", str(CAP)])

    assert a_kept_its_id(load_tracks(tmp_path / "off.jsonl"), returned)
    assert not a_kept_its_id(load_tracks(tmp_path / "on.jsonl"), returned)
