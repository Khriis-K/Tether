"""The search gate rejects a detection whose size doesn't match the searching track's.

Center distance ignores size, so without this a searching track behind someone
takes the much bigger box of the person walking in front of it (MOT17-09, #35).
The check is against the track's last observed box: a coasted prediction's size
drifts with the size velocity. Scripted, stationary objects unless noted.
"""

import pytest

from tether.formats import Detection
from tether.matching import MIN_IOU, iou
from tether.tracker import SEARCH_GATE_MAX_SIZE_RATIO, Tracker

SIZE = 20.0
X, Y = 200.0, 100.0
CONFIRM = 3


def box_at(cx, cy=Y + SIZE / 2, size=SIZE):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def run(frames):
    tracker = Tracker(confirm_frames=CONFIRM)  # the default gate and cap
    return [tracker.update(dets) for dets in frames]


def kept_its_id(outputs, returned):
    searching = [t for t in outputs[-2] if t.status == "searching"]
    assert len(searching) == 1
    at_returned = [t for t in outputs[-1] if t.box == returned and t.frames_since_seen == 0]
    assert len(at_returned) == 1
    return at_returned[0].id == searching[0].id


def confirmed_then_missing(missed, box):
    return [[det(box)] for _ in range(CONFIRM + 2)] + [[] for _ in range(missed)]


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_detection_much_bigger_or_smaller_starts_a_new_track(scale):
    # Same center, so well inside the gate, but not the same object.
    assert scale > SEARCH_GATE_MAX_SIZE_RATIO or 1 / scale > SEARCH_GATE_MAX_SIZE_RATIO
    old = box_at(X)
    returned = box_at(X, size=SIZE * scale)
    assert iou(old, returned) < MIN_IOU

    assert not kept_its_id(run(confirmed_then_missing(1, old) + [[det(returned)]]), returned)


def test_modest_size_change_still_rematches():
    old = box_at(X)
    returned = box_at(X + 1.25 * SIZE, size=SIZE * 1.25)  # no overlap, inside the gate
    assert 1.25 < SEARCH_GATE_MAX_SIZE_RATIO
    assert iou(old, returned) < MIN_IOU

    assert kept_its_id(run(confirmed_then_missing(1, old) + [[det(returned)]]), returned)


def test_size_is_compared_to_the_last_observed_box_not_the_coasted_prediction():
    # An object growing every frame: coasting, its predicted box keeps growing,
    # so by the time it's seen again at its last size the prediction is far bigger.
    growing = [box_at(X, size=SIZE + 4.0 * i) for i in range(25)]
    last_size = SIZE + 4.0 * 24
    returned = box_at(X + last_size / 2, size=last_size)  # half a box over, so IoU can't match it
    frames = [[det(b)] for b in growing] + [[] for _ in range(29)] + [[det(returned)]]

    outputs = run(frames)

    predicted = outputs[-2][0].predicted_box
    assert (predicted[3] - predicted[1]) / last_size > SEARCH_GATE_MAX_SIZE_RATIO
    assert iou(predicted, returned) < MIN_IOU
    assert kept_its_id(outputs, returned)
