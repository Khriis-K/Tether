"""The single-target prior (#34): a diagnostic upper bound, off by default.

With single_target on, if exactly one track is searching and none is tracked,
that track matches the nearest high detection anywhere in the frame, with no
IoU, gate, cap or size limit. It never fires while another confirmed track
exists. Scripted, stationary objects: the prediction stays where the box was.
"""

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import Tracker

SIZE = 20.0
X, Y = 200.0, 100.0
CONFIRM = 3
FAR = 30.0  # box sizes: past every default gate and merge reach


def box_at(x, y=Y, size=SIZE):
    return (x, y, x + size, y + size)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def run(frames, **kwargs):
    tracker = Tracker(confirm_frames=CONFIRM, **kwargs)
    return [tracker.update(dets) for dets in frames]


def missed_then_back(returned, others=()):
    """Object A at X, confirmed, missed for one frame, then detected only at `returned`.

    `others` are the boxes of further objects, detected on every frame up to the miss.
    """
    seen = [det(box_at(X))] + [det(b) for b in others]
    return [seen for _ in range(CONFIRM + 2)] + [[], [det(b) for b in returned]]


def holder(frame, box):
    [track] = [t for t in frame if t.box == box and t.frames_since_seen == 0]
    return track


def a_id(outputs):
    [a] = [t for t in outputs[CONFIRM] if t.box == box_at(X)]
    return a.id


def test_off_by_default_a_far_return_starts_a_new_track():
    returned = box_at(X + FAR * SIZE)

    outputs = run(missed_then_back([returned]))

    new = holder(outputs[-1], returned)
    assert new.id != a_id(outputs)
    assert new.status == "tentative"


def test_on_the_only_searching_track_takes_a_far_detection_of_any_size():
    returned = box_at(X + FAR * SIZE, size=3 * SIZE)

    outputs = run(missed_then_back([returned]), single_target=True)

    kept = holder(outputs[-1], returned)
    assert kept.id == a_id(outputs)
    assert kept.status == "tracked"
    assert len(outputs[-1]) == 1


def test_on_it_takes_the_nearest_detection():
    near, farther = box_at(X + FAR * SIZE), box_at(X - (FAR + 10) * SIZE)

    outputs = run(missed_then_back([farther, near]), single_target=True)

    assert holder(outputs[-1], near).id == a_id(outputs)
    assert holder(outputs[-1], farther).status == "tentative"


def test_on_it_never_fires_with_two_searching_tracks():
    other = box_at(X, y=Y + 10 * SIZE)
    returned = box_at(X + FAR * SIZE)

    outputs = run(missed_then_back([returned], others=[other]), single_target=True)

    assert [t.status for t in outputs[-2]] == ["searching", "searching"]
    assert holder(outputs[-1], returned).status == "tentative"


def test_on_it_never_fires_while_another_track_is_tracked():
    other = box_at(X, y=Y + 10 * SIZE)
    returned = box_at(X + FAR * SIZE)
    frames = missed_then_back([returned], others=[other])
    frames[-2] = [det(other)]
    frames[-1] = frames[-1] + [det(other)]

    outputs = run(frames, single_target=True)

    assert sorted(t.status for t in outputs[-2]) == ["searching", "tracked"]
    assert holder(outputs[-1], returned).status == "tentative"
    assert holder(outputs[-1], other).status == "tracked"


def test_cli_flag_turns_it_on(tmp_path):
    returned = box_at(X + FAR * SIZE)
    dets_path = tmp_path / "dets.jsonl"
    save_detections(dets_path, missed_then_back([returned]))

    def returned_holder(*flags):
        tracks_path = tmp_path / "tracks.jsonl"
        assert main([str(dets_path), str(tracks_path), "--confirm-frames", str(CONFIRM), *flags]) == 0
        frames = load_tracks(tracks_path)
        return holder(frames[-1], returned).id == a_id(frames)

    assert not returned_holder()
    assert returned_holder("--single-target")
