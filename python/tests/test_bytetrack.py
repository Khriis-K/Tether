import pytest

pytest.importorskip("ultralytics")

from tether.bytetrack import main, track_detections  # noqa: E402
from tether.formats import Detection, load_tracks, save_detections  # noqa: E402

PERSON = 0


def walker(f, score=0.9, cls=PERSON):
    """A person walking right at 4 px a frame."""
    return Detection(box=(100.0 + 4 * f, 50.0, 140.0 + 4 * f, 150.0), score=score, cls=cls)


def ids(frame):
    return [t.id for t in frame]


def test_a_steady_walker_keeps_one_id():
    frames = track_detections([[walker(f)] for f in range(20)])

    assert len(frames) == 20
    assert {i for frame in frames for i in ids(frame)} == {1}
    assert all(t.status == "tracked" and t.cls == PERSON for frame in frames for t in frame)


def test_low_score_detections_keep_a_track_going_but_never_start_one():
    # ByteTrack's second pass: 0.15 is under the high threshold but over the low one.
    stranger = Detection(box=(500.0, 50.0, 540.0, 150.0), score=0.15, cls=PERSON)
    frames = [[walker(f)] for f in range(10)] + [[walker(f, score=0.15), stranger] for f in range(10, 20)]

    out = track_detections(frames)

    assert all(ids(frame) == [1] for frame in out)


def test_other_classes_are_ignored():
    frames = [[walker(f), walker(f, cls=2)] for f in range(5)]

    out = track_detections(frames)

    assert all(len(frame) == 1 and frame[0].cls == PERSON for frame in out)


def test_a_lost_track_is_dropped_from_the_output_and_keeps_its_id_when_found():
    frames = [[walker(f)] for f in range(10)] + [[] for _ in range(5)] + [[walker(f)] for f in range(15, 20)]

    out = track_detections(frames)

    assert all(frame == [] for frame in out[10:15])
    assert all(ids(frame) == [1] for frame in out[15:])


def test_cli_writes_one_track_frame_per_detection_frame(tmp_path):
    dets_path, tracks_path = tmp_path / "dets.jsonl", tmp_path / "tracks.jsonl"
    save_detections(dets_path, [[walker(0)], [], [walker(2)]])

    assert main([str(dets_path), str(tracks_path)]) == 0

    assert len(load_tracks(tracks_path)) == 3


def test_without_the_low_score_pass_a_weak_walker_is_lost():
    frames = [[walker(f)] for f in range(10)] + [[walker(f, score=0.15)] for f in range(10, 20)]

    out = track_detections(frames, low_score_pass=False)

    assert all(frame == [] for frame in out[10:])
