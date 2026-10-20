"""A tentative track survives one missed frame and is confirmed on net hits (hits minus misses).

Without this, a tentative track needs `confirm_frames` hits in a row, so at 20%
deletion one deleted box in its first frames drops it and the next detection
starts yet another ID. Scripted, stationary objects.
"""

import pytest

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import Tracker

SIZE = 20.0


def box_at(x, y=50.0, size=SIZE):
    return (x, y, x + size, y + size)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def run(frames, **settings):
    tracker = Tracker(**settings)
    return [tracker.update(dets) for dets in frames]


def pattern(hits, x=100.0):
    """One object at `x`, detected on the frames marked 1 in `hits`."""
    return [[det(box_at(x))] if h else [] for h in hits]


def test_tentative_track_survives_one_missed_frame_and_keeps_its_id():
    outputs = run(pattern([1, 1, 0, 1]), confirm_frames=3)

    assert [t.status for t in outputs[2]] == ["tentative"]
    assert outputs[2][0].frames_since_seen == 1
    assert len({tracks[0].id for tracks in outputs}) == 1


def test_tentative_track_is_dropped_on_its_second_missed_frame_in_a_row():
    outputs = run(pattern([1, 1, 0, 0, 1]), confirm_frames=3)

    assert outputs[3] == []
    assert outputs[4][0].id != outputs[0][0].id


@pytest.mark.parametrize("confirm_frames", [2, 3, 5])
def test_a_missed_frame_takes_back_one_hit_before_confirming(confirm_frames):
    # Hit, miss, then hits: net hits reach confirm_frames one hit later than without the miss.
    outputs = run(pattern([1, 0] + [1] * (confirm_frames + 2)), confirm_frames=confirm_frames)

    statuses = [tracks[0].status for tracks in outputs]
    # Net hits after frame f (f >= 1) are f - 1; confirmed once they reach confirm_frames.
    confirmed_at = confirm_frames + 1
    assert statuses[:confirmed_at] == ["tentative"] * confirmed_at
    assert statuses[confirmed_at] == "tracked"
    assert len({tracks[0].id for tracks in outputs}) == 1


def test_one_off_false_detection_is_never_confirmed():
    frames = [[det(box_at(0.0, y=300.0))] for _ in range(10)]
    frames[3].append(det(box_at(400.0)))

    outputs = run(frames, confirm_frames=3)

    blip = {t.id for t in outputs[3]} - {outputs[0][0].id}
    assert len(blip) == 1
    blip_tracks = [t for tracks in outputs for t in tracks if t.id in blip]
    assert all(t.status == "tentative" for t in blip_tracks)
    assert all(t.id not in blip for tracks in outputs[5:] for t in tracks)


def test_detection_that_flickers_every_other_frame_is_never_confirmed():
    outputs = run(pattern([1, 0] * 10), confirm_frames=3)

    assert all(t.status == "tentative" for tracks in outputs for t in tracks)


def test_zero_tentative_misses_drops_a_tentative_track_on_its_first_miss():
    outputs = run(pattern([1, 1, 0, 1]), confirm_frames=3, tentative_misses=0)

    assert outputs[2] == []
    assert outputs[3][0].id != outputs[0][0].id


def test_a_tentative_track_is_never_merged_into_a_younger_track():
    # T is hit twice, then flickers at x=100 (net hits stay at 1 or 2), so it outlives
    # O, born at x=160 on frame 2 and confirmed. (Two hits first, or T's one-hit
    # distance match would take O's first detection.) O stops being detected on frame
    # 7, as T is detected on every frame: T confirms on frame 8, 3 box sizes from O's
    # prediction, outside O's gate but inside its merge reach. O is younger, so T
    # keeps its own ID.
    t_hits = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    frames = [([det(box_at(100.0))] if h else []) + ([det(box_at(160.0))] if 2 <= f <= 6 else [])
              for f, h in enumerate(t_hits)]

    outputs = run(frames, confirm_frames=3)

    t_id = next(t.id for t in outputs[0])
    o_id = next(t.id for t in outputs[5] if t.box == box_at(160.0))
    assert o_id > t_id
    holder = next(t for t in outputs[8] if t.box == box_at(100.0))
    assert holder.id == t_id
    assert holder.status == "tracked"
    assert next(t for t in outputs[8] if t.id == o_id).status == "searching"


def test_a_duplicate_that_missed_a_frame_is_merged_only_once_its_net_hits_confirm_it():
    # A is confirmed at x=400, then the only detection is 6 box sizes away (outside
    # A's gate, inside its merge reach), hit, hit, miss, hit, hit. The duplicate has
    # 3 hits on the 4th of those frames but net hits of 2, so it isn't merged until the 5th.
    before = [[det(box_at(400.0))] for _ in range(5)]
    jump = box_at(400.0 + 6 * SIZE)
    outputs = run(before + pattern([1, 1, 0, 1, 1], x=jump[0]), confirm_frames=3)

    a_id = outputs[4][0].id
    at_jump = [t for t in outputs[8] if t.box == jump]
    assert [(t.status, t.id != a_id) for t in at_jump] == [("tentative", True)]
    assert [(t.id, t.status) for t in outputs[9] if t.box == jump] == [(a_id, "tracked")]


def test_cli_passes_tentative_misses_to_the_tracker(tmp_path):
    dets_path = tmp_path / "dets.jsonl"
    tracks_path = tmp_path / "tracks.jsonl"
    save_detections(dets_path, pattern([1, 1, 0, 1]))

    assert main([str(dets_path), str(tracks_path), "--tentative-misses", "0"]) == 0

    frames = load_tracks(tracks_path)
    assert frames[2] == []
