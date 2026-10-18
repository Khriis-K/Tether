"""Track lifecycle: tentative -> tracked -> searching -> removed. Scripted detections in, tracks out."""

import pytest

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import Tracker

SIZE = 20.0


def box_at(x, y=50.0, size=SIZE):
    return (x, y, x + size, y + size)


def det(box, cls=0):
    return Detection(box=box, score=0.9, cls=cls)


def run(frames, **settings):
    tracker = Tracker(**settings)
    return [tracker.update(dets) for dets in frames]


def moving(f, speed=3.0, start_x=0.0, y=50.0):
    return box_at(start_x + speed * f, y=y)


def with_gap(n_frames, gap, speed=3.0):
    """One object moving steadily, undetected on the frames in `gap`."""
    return [[] if f in gap else [det(moving(f, speed))] for f in range(n_frames)]


def by_id(tracks):
    return {t.id: t for t in tracks}


def test_new_detection_starts_as_a_tentative_track():
    tracks = run([[det(box_at(0.0))]], confirm_frames=3)[0]

    assert [t.status for t in tracks] == ["tentative"]


@pytest.mark.parametrize("confirm_frames", [1, 2, 3, 5])
def test_track_is_confirmed_after_the_confirm_threshold_in_frames(confirm_frames):
    outputs = run([[det(moving(f))] for f in range(8)], confirm_frames=confirm_frames)

    statuses = [tracks[0].status for tracks in outputs]
    assert statuses == ["tentative"] * (confirm_frames - 1) + ["tracked"] * (9 - confirm_frames)
    assert len({tracks[0].id for tracks in outputs}) == 1


def test_one_frame_false_detection_never_becomes_a_confirmed_track():
    # A real object throughout, plus a blip far away on frame 5 only.
    frames = [[det(moving(f))] for f in range(15)]
    frames[5].append(det(box_at(400.0, y=300.0)))

    outputs = run(frames, confirm_frames=3, max_missing_frames=30)

    real_id = outputs[0][0].id
    blip_ids = {t.id for t in outputs[5]} - {real_id}
    assert len(blip_ids) == 1
    blip_id = blip_ids.pop()
    assert by_id(outputs[5])[blip_id].status == "tentative"
    for tracks in outputs[6:]:
        assert blip_id not in by_id(tracks)
    assert all(t.status in ("tentative", "tracked") for tracks in outputs for t in tracks
               if t.id == real_id)


def test_tentative_track_that_misses_a_frame_is_dropped_before_confirming():
    outputs = run([[det(box_at(0.0))], [det(box_at(0.0))], [], [det(box_at(0.0))]],
                  confirm_frames=3, max_missing_frames=30)

    assert outputs[2] == []
    assert outputs[3][0].id != outputs[0][0].id
    assert outputs[3][0].status == "tentative"


def test_confirmed_track_that_loses_its_detection_is_searching_and_counts_missed_frames():
    outputs = run(with_gap(15, gap=range(10, 15)), confirm_frames=3, max_missing_frames=30)

    track_id = outputs[0][0].id
    for f, k in zip(range(10, 15), range(1, 6)):
        assert [t.id for t in outputs[f]] == [track_id]
        assert outputs[f][0].status == "searching"
        assert outputs[f][0].frames_since_seen == k


def test_searching_track_keeps_predicting_along_its_motion_and_keeps_its_last_seen_box():
    outputs = run(with_gap(16, gap=range(10, 16)), confirm_frames=3, max_missing_frames=30)

    for f in range(10, 16):
        track = outputs[f][0]
        assert track.box == moving(9)
        # Predicted box for the next frame follows the object's 3 px/frame.
        assert track.predicted_box == pytest.approx(moving(f + 1), abs=1.0)


@pytest.mark.parametrize("gap_length", [1, 5, 12])
def test_object_that_disappears_for_fewer_frames_than_the_limit_keeps_its_id(gap_length):
    gap = range(10, 10 + gap_length)
    outputs = run(with_gap(10 + gap_length + 5, gap), confirm_frames=3, max_missing_frames=12)

    track_id = outputs[0][0].id
    for tracks in outputs:
        assert [t.id for t in tracks] == [track_id]
    back = outputs[10 + gap_length][0]
    assert back.status == "tracked"
    assert back.frames_since_seen == 0
    assert back.box == moving(10 + gap_length)


def test_stationary_object_that_disappears_and_returns_keeps_its_id():
    frames = [[det(box_at(100.0))] for _ in range(10)] + [[]] * 8 + \
             [[det(box_at(100.0))] for _ in range(5)]

    outputs = run(frames, confirm_frames=3, max_missing_frames=12)

    assert {t.id for tracks in outputs for t in tracks} == {outputs[0][0].id}


@pytest.mark.parametrize("max_missing_frames", [1, 5, 12])
def test_track_is_removed_once_missing_longer_than_the_limit(max_missing_frames):
    gap = range(10, 10 + max_missing_frames + 1)
    outputs = run(with_gap(10 + max_missing_frames + 3, gap),
                  confirm_frames=3, max_missing_frames=max_missing_frames)

    track_id = outputs[0][0].id
    last_searching = 10 + max_missing_frames - 1
    assert outputs[last_searching][0].id == track_id
    assert outputs[last_searching][0].frames_since_seen == max_missing_frames
    assert outputs[last_searching + 1] == []
    # The object comes back, but its old track is gone: it's a new object now.
    returned = outputs[gap.stop][0]
    assert returned.id != track_id
    assert returned.status == "tentative"


def test_two_objects_crossing_while_one_hides_the_other_do_not_swap_ids():
    # Same lane, 6 px/frame toward each other, meeting around frame 20. While
    # they overlap, only the front one is detected; the hidden one is searching.
    def rightward(f):
        return box_at(6.0 * f, y=50.0)

    def leftward(f):
        return box_at(240.0 - 6.0 * f, y=54.0)

    hidden = range(17, 24)
    frames = [[det(rightward(f))] + ([] if f in hidden else [det(leftward(f))])
              for f in range(40)]

    outputs = run(frames, confirm_frames=3, max_missing_frames=30)

    def id_with_box(tracks, box):
        return next(t.id for t in tracks if t.box == box)

    right_id = id_with_box(outputs[0], rightward(0))
    left_id = id_with_box(outputs[0], leftward(0))
    assert right_id != left_id
    for f, tracks in enumerate(outputs):
        assert id_with_box(tracks, rightward(f)) == right_id
        if f not in hidden:
            assert id_with_box(tracks, leftward(f)) == left_id
    assert all(by_id(outputs[f])[left_id].status == "searching" for f in hidden)


def test_default_settings_confirm_and_remove_tracks():
    tracker = Tracker()
    statuses = [tracker.update([det(box_at(0.0))])[0].status for _ in range(10)]
    assert statuses[0] == "tentative"
    assert statuses[-1] == "tracked"

    outputs = [tracker.update([]) for _ in range(1000)]
    assert outputs[0][0].status == "searching"
    assert outputs[-1] == []


def test_cli_passes_lifecycle_settings_to_the_tracker(tmp_path):
    dets_path = tmp_path / "dets.jsonl"
    tracks_path = tmp_path / "tracks.jsonl"
    save_detections(dets_path, [[det(box_at(0.0))], [], [], []])

    assert main([str(dets_path), str(tracks_path),
                 "--confirm-frames", "1", "--max-missing-frames", "2"]) == 0

    frames = load_tracks(tracks_path)
    assert [[t.status for t in tracks] for tracks in frames] == \
           [["tracked"], ["searching"], ["searching"], []]
