"""Merging a duplicate into the searching track it came from.

When a tentative track is confirmed and an older confirmed track missed this
frame, is about the same size, and has its prediction within its merge reach
(`merge_reach + merge_reach_growth * frames_since_seen` box sizes, capped like
the search gate), the older track takes over the new one and keeps its ID.
Scripted, stationary objects: the prediction stays where the box was.
"""

import pytest

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import SEARCH_GATE_MAX_SIZE_RATIO, Tracker

SIZE = 20.0
X, Y = 400.0, 300.0
CONFIRM = 3
JUMP = 6.0  # box sizes: far outside the search gate, inside the default merge reach


def box_at(x, y=Y, size=SIZE):
    return (x, y, x + size, y + size)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def jumped(offset, size=SIZE, others=(), after=CONFIRM):
    """Object A at X, confirmed, then detected only at X + offset box sizes for `after` frames.

    `others` are boxes detected on every frame.
    """
    extra = [det(b) for b in others]
    before = [[det(box_at(X))] + extra for _ in range(CONFIRM + 2)]
    return before + [[det(box_at(X + offset * SIZE, size=size))] + extra for _ in range(after)]


def run(frames, **kwargs):
    tracker = Tracker(confirm_frames=CONFIRM, **kwargs)
    return [tracker.update(dets) for dets in frames]


def a_id(outputs):
    (a,) = [t for t in outputs[CONFIRM + 1] if t.box == box_at(X)]
    return a.id


def at_jump(outputs, offset, size=SIZE, frame=-1):
    (track,) = [t for t in outputs[frame] if t.box == box_at(X + offset * SIZE, size=size)]
    return track


def test_duplicate_born_while_the_original_searches_ends_with_the_original_id():
    outputs = run(jumped(JUMP))

    assert at_jump(outputs, JUMP).id == a_id(outputs)
    assert at_jump(outputs, JUMP).status == "tracked"
    assert len(outputs[-1]) == 1  # the younger ID is gone


def test_merged_track_keeps_following_the_object():
    # The older track takes over the new track's motion; updating its own coasted
    # filter with a detection 6 box sizes away would leave it off the object.
    outputs = run(jumped(JUMP, after=CONFIRM + 5))

    assert all(at_jump(outputs, JUMP, frame=f).id == a_id(outputs) for f in range(-5, 0))


def test_two_objects_side_by_side_tracked_the_whole_time_are_never_merged():
    b = box_at(X + 3.0 * SIZE)
    frames = [[det(box_at(X))] for _ in range(CONFIRM + 2)]
    frames += [[det(box_at(X)), det(b)] for _ in range(CONFIRM + 5)]

    outputs = run(frames)

    assert len({t.id for t in outputs[-1] if t.status == "tracked"}) == 2


@pytest.mark.parametrize("scale", [SEARCH_GATE_MAX_SIZE_RATIO * 1.2, 1 / (SEARCH_GATE_MAX_SIZE_RATIO * 1.2)])
def test_a_much_bigger_or_smaller_box_is_never_merged(scale):
    outputs = run(jumped(JUMP, size=SIZE * scale))

    assert at_jump(outputs, JUMP, size=SIZE * scale).id != a_id(outputs)


def test_a_slightly_different_size_is_still_merged():
    scale = SEARCH_GATE_MAX_SIZE_RATIO * 0.9
    outputs = run(jumped(JUMP, size=SIZE * scale))

    assert at_jump(outputs, JUMP, size=SIZE * scale).id == a_id(outputs)


@pytest.mark.parametrize("cap, merged", [(0.5, False), (float("inf"), True)])
def test_a_nearby_confirmed_track_caps_the_reach(cap, merged):
    # C sits 2 box sizes from A on the other side, so the cap is 1 box size.
    c = box_at(X - 2.0 * SIZE)

    outputs = run(jumped(JUMP, others=[c]), search_gate_cap=cap)

    assert (at_jump(outputs, JUMP).id == a_id(outputs)) == merged


@pytest.mark.parametrize("growth, merged", [(1.0, True), (0.0, False)])
def test_the_reach_grows_with_frames_searched(growth, merged):
    # The duplicate is confirmed on A's third missed frame (2 counted before it),
    # so the reach is 2 + 2 * growth: 4 reaches 3.5, 2 doesn't. Gate off.
    outputs = run(jumped(3.5), search_gate=0.0, search_gate_growth=0.0,
                  merge_reach=2.0, merge_reach_growth=growth)

    assert (at_jump(outputs, 3.5).id == a_id(outputs)) == merged


def test_a_duplicate_beyond_the_default_reach_keeps_its_own_id():
    # Confirmed with 2 missed frames counted: the default reach is 10 + 2 = 12.
    outputs = run(jumped(13.0))

    assert at_jump(outputs, 13.0).id != a_id(outputs)


def test_cli_passes_the_merge_reach_to_the_tracker(tmp_path):
    save_detections(tmp_path / "dets.jsonl", jumped(JUMP))
    args = ["--confirm-frames", str(CONFIRM)]

    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "on.jsonl"), *args])
    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "off.jsonl"), *args,
          "--merge-reach", "0", "--merge-reach-growth", "0"])

    on, off = load_tracks(tmp_path / "on.jsonl"), load_tracks(tmp_path / "off.jsonl")
    assert at_jump(on, JUMP).id == a_id(on)
    assert at_jump(off, JUMP).id != a_id(off)
