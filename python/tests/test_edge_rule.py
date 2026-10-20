"""The edge rule: a track whose last observed box touched the frame edge most likely left the scene.

Its ID is never handed to anyone else by the search gate or the duplicate merge;
IoU still matches it, so an object standing at the edge keeps its ID through a
missed frame. A box touches the edge when it's within EDGE_MARGIN of its own
height of it. The tracker needs the frame size for this; without one the rule is off.
"""

import sys

import cv2
import numpy as np
import pytest

from tether import video
from tether.antiuav import evaluate as antiuav_evaluate
from tether.antiuav import main as antiuav_main
from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.mot17 import evaluate as mot17_evaluate
from tether.tracker import EDGE_MARGIN, Tracker

# --- The scripted webcam scene from #37 ---

W, H = 1920, 1080  # 30 fps webcam
PW, PH = 300, 800  # a person close to the camera
SPEED = 40  # px per frame, walking left


def person(x):
    """A person's box at left edge `x`, clipped to the frame as a detector gives it."""
    return Detection(box=(max(0.0, x), 200.0, min(float(W), x + PW), 200.0 + PH), score=0.9, cls=0)


def walk_out_walk_in(gap):
    """A walks out the left edge; `gap` empty frames later, B walks in at the right edge."""
    frames, x = [], 800.0
    while x + PW > 60:  # A's clipped box shrinks in width only, and stays at the edge
        frames.append([person(x)])
        x -= SPEED
    frames += [[] for _ in range(gap)]
    x = W - 60.0
    for _ in range(20):
        frames.append([person(x)])
        x -= SPEED
    return frames


def a_and_b_ids(outputs):
    (a,) = outputs[5]
    b = {t.id for t in outputs[-1] if t.status == "tracked"}
    return a.id, b


@pytest.mark.parametrize("gap", [0, 1, 5, 15, 29])
def test_a_person_who_walks_out_at_the_edge_never_gives_their_id_to_a_newcomer(gap):
    tracker = Tracker(frame_size=(W, H))

    a, b = a_and_b_ids([tracker.update(dets) for dets in walk_out_walk_in(gap)])

    assert b and a not in b


@pytest.mark.parametrize("gap", [5, 15, 29])
def test_the_gate_alone_never_gives_the_id_away_at_the_edge(gap):
    # Merge off: before the edge rule the gate alone swapped for every gap of 5-29 frames.
    tracker = Tracker(frame_size=(W, H), merge_reach=0.0, merge_reach_growth=0.0)

    a, b = a_and_b_ids([tracker.update(dets) for dets in walk_out_walk_in(gap)])

    assert b and a not in b


def test_the_merge_alone_never_gives_the_id_away_at_the_edge():
    # Gap 0, gate off: the newcomer arrives before A starts searching, which only the merge reaches.
    tracker = Tracker(frame_size=(W, H), search_gate=0.0, search_gate_growth=0.0)

    a, b = a_and_b_ids([tracker.update(dets) for dets in walk_out_walk_in(0)])

    assert b and a not in b


def test_without_a_frame_size_the_rule_is_off():
    tracker = Tracker()

    a, b = a_and_b_ids([tracker.update(dets) for dets in walk_out_walk_in(5)])

    assert a in b  # the swap #37 is about


# --- A small object, mid-frame or at one edge ---

SIZE = 20.0
FW, FH = 640, 480
CONFIRM = 3


def box_at(x, y):
    return (x, y, x + SIZE, y + SIZE)


def det(box):
    return Detection(box=box, score=0.9, cls=0)


def missed_then_back(start, back, missed=1):
    """An object at `start`, confirmed, missed for `missed` frames, then detected at `back`."""
    frames = [[det(start)] for _ in range(CONFIRM + 2)] + [[] for _ in range(missed)]
    return frames + [[det(back)] for _ in range(CONFIRM)]


def kept_its_id(frames):
    tracker = Tracker(confirm_frames=CONFIRM, frame_size=(FW, FH))
    outputs = [tracker.update(dets) for dets in frames]
    (original,) = outputs[CONFIRM + 1]
    return [t.id for t in outputs[-1] if t.status == "tracked"] == [original.id]


MID = (300.0, 200.0)


def test_an_object_lost_mid_frame_is_still_rematched_by_the_gate():
    # 1.25 box sizes: no overlap, so only the gate can match it.
    assert kept_its_id(missed_then_back(box_at(*MID), box_at(MID[0] + 1.25 * SIZE, MID[1])))


def test_an_object_lost_mid_frame_is_still_merged():
    # 6 box sizes: past the gate, inside the merge reach.
    assert kept_its_id(missed_then_back(box_at(*MID), box_at(MID[0] + 6 * SIZE, MID[1]), missed=0))


EDGES = {  # a box touching each edge, and the direction back toward the middle
    "left": (box_at(0.0, 200.0), (1, 0)),
    "right": (box_at(FW - SIZE, 200.0), (-1, 0)),
    "top": (box_at(300.0, 0.0), (0, 1)),
    "bottom": (box_at(300.0, FH - SIZE), (0, -1)),
}


def inward(edge, box_sizes):
    """The box touching `edge`, moved `box_sizes` toward the middle."""
    (x1, y1, _, _), (dx, dy) = EDGES[edge]
    return box_at(x1 + dx * box_sizes * SIZE, y1 + dy * box_sizes * SIZE)


@pytest.mark.parametrize("edge", EDGES)
def test_a_track_last_seen_at_any_edge_is_not_rematched_by_the_gate(edge):
    # 1.25 box sizes: inside the gate, no overlap.
    assert not kept_its_id(missed_then_back(inward(edge, 0), inward(edge, 1.25)))


@pytest.mark.parametrize("edge", EDGES)
def test_a_track_last_seen_at_any_edge_is_not_merged(edge):
    # 5 box sizes: past the gate, inside the merge reach.
    assert not kept_its_id(missed_then_back(inward(edge, 0), inward(edge, 5), missed=0))


def test_a_box_partly_outside_the_frame_touches_the_edge():
    # Ground-truth boxes, as Anti-UAV feeds them, aren't clipped to the frame.
    outside = (-5.0, 200.0, -5.0 + SIZE, 200.0 + SIZE)

    assert not kept_its_id(missed_then_back(outside, box_at(1.0 * SIZE, 200.0)))


@pytest.mark.parametrize("gap, touches", [(EDGE_MARGIN * 0.9, True), (EDGE_MARGIN * 1.1, False)])
def test_within_the_margin_of_the_box_height_counts_as_touching(gap, touches):
    start = box_at(gap * SIZE, 200.0)
    back = box_at(gap * SIZE + 1.25 * SIZE, 200.0)

    assert kept_its_id(missed_then_back(start, back)) != touches


def test_the_margin_is_in_box_heights_not_widths():
    # A box clipped at the edge loses width but keeps its height: 4 px wide, 20 tall,
    # 0.8 px from the edge. That's within 0.05 heights (1 px) but not 0.05 widths (0.2 px).
    start = (0.8, 200.0, 4.8, 220.0)
    back = (0.8 + 1.0 * SIZE, 200.0, 4.8 + 1.0 * SIZE, 220.0)

    assert not kept_its_id(missed_then_back(start, back))


def test_a_track_at_the_edge_still_keeps_its_id_by_iou():
    # Someone standing at the edge, missed for a frame, then back in about the same place.
    at_edge, _ = EDGES["left"]
    nudged = box_at(0.2 * SIZE, 200.0)

    assert kept_its_id(missed_then_back(at_edge, nudged))


def test_only_the_last_observed_box_counts_not_the_prediction():
    # Moving left 20 px a frame, last seen 60 px from the edge: after 3 missed
    # frames the prediction has coasted past the edge, but the last observed box
    # is well inside it, so the gate still re-matches the object at the edge.
    seen = [box_at(60.0 + 20.0 * (11 - f), 200.0) for f in range(12)]
    frames = [[det(b)] for b in seen] + [[], [], [], [det(box_at(0.0, 200.0))]]
    tracker = Tracker(confirm_frames=CONFIRM, frame_size=(FW, FH))

    outputs = [tracker.update(dets) for dets in frames]

    (searching,) = outputs[-2]
    assert searching.predicted_box[0] <= 0  # the prediction is at the edge
    assert [t.id for t in outputs[-1]] == [searching.id]
    assert outputs[-1][0].status == "tracked"


# --- Passing the frame size in ---

def test_cli_passes_the_frame_size_to_the_tracker(tmp_path):
    save_detections(tmp_path / "dets.jsonl", walk_out_walk_in(5))

    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "off.jsonl")])
    main([str(tmp_path / "dets.jsonl"), str(tmp_path / "on.jsonl"), "--frame-size", str(W), str(H)])

    a, b = a_and_b_ids(load_tracks(tmp_path / "off.jsonl"))
    assert a in b
    a, b = a_and_b_ids(load_tracks(tmp_path / "on.jsonl"))
    assert b and a not in b


def test_opening_a_video_gives_its_frame_size(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()

    frames, size = video.open_video(str(path))

    assert size == (64, 48)
    assert len(list(frames)) == 3


ARGS_LOGGER = (
    "import sys\n"
    "*size, dets, tracks = sys.argv[1:]\n"
    "open(sys.argv[0] + '.log', 'a', encoding='utf-8').write(' '.join(size) + '\\n')\n"
    "lines = open(dets, encoding='utf-8').read().splitlines()\n"
    "with open(tracks, 'w', encoding='utf-8', newline='\\n') as f:\n"
    "    for i, _ in enumerate(lines):\n"
    "        f.write('{\"frame\": %d, \"tracks\": []}\\n' % i)\n")


def test_mot17_fills_in_each_sequences_frame_size(tmp_path):
    from test_mot17 import GroundTruthDetector, make_sequence
    root = tmp_path / "MOT17"
    make_sequence(root / "train", "MOT17-02-FRCNN", with_images=True)  # 320x240
    script = tmp_path / "args.py"
    script.write_text(ARGS_LOGGER, encoding="utf-8")

    mot17_evaluate(root, tmp_path / "work", f'"{sys.executable}" "{script}" {{width}} {{height}}', "args",
                   GroundTruthDetector())

    assert (tmp_path / "args.py.log").read_text(encoding="utf-8") == "320 240\n"


def test_antiuav_fills_in_each_clips_frame_size(tmp_path):
    from test_antiuav import write_clip
    root = tmp_path / "Anti-UAV300"
    write_clip(root / "test", "c1", [box_at(10.0, 10.0)] * 5)  # 64x48 video
    script = tmp_path / "args.py"
    script.write_text(ARGS_LOGGER, encoding="utf-8")

    antiuav_evaluate(root, tmp_path / "work", f'"{sys.executable}" "{script}" {{width}} {{height}}', "args",
                     drop_rates=[0.0], gap_seconds=[])

    assert (tmp_path / "args.py.log").read_text(encoding="utf-8") == "64 48\n"


def test_antiuav_runs_tether_with_the_frame_size_by_default(tmp_path):
    from test_antiuav import write_clip
    # A 10 px drone at the right edge of a 64x48 clip, gone for 3 frames, then
    # a same-size one mid-frame: without the edge rule the merge hands it A's ID.
    a, b = (54.0, 20.0, 64.0, 30.0), (20.0, 20.0, 30.0, 30.0)
    root = tmp_path / "Anti-UAV300"
    write_clip(root / "test", "c1", [a] * 10 + [None] * 3 + [b] * 18)
    work = tmp_path / "work"

    antiuav_main([str(root), "--work-dir", str(work), "--drop-rates", "0", "--gap-seconds"])

    frames = load_tracks(work / "trackers" / "tether" / "drop-0.0-seed0" / "c1.jsonl")
    (at_a,) = [t.id for t in frames[9] if t.status == "tracked"]
    assert at_a not in {t.id for t in frames[-1] if t.status == "tracked"}
