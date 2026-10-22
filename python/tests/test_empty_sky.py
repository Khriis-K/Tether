"""The empty-sky set (#45): Anti-UAV frames with the drone labeled absent AND no labeled drone
nearby, so a detection there is an alarm on empty sky rather than on a drone half out of view
that the labels call absent (#43, #44). The rule reads only labels and frame indices."""

import json

import pytest

from tether.antiuav import Clip
from tether.empty_sky import empty_sky_frames, empty_sky_split, load_empty_sky, main, to_ranges

D = (0.0, 0.0, 10.0, 10.0)


def test_an_absent_frame_more_than_the_margin_from_any_labeled_drone_is_empty_sky():
    boxes = [None] * 3 + [D] + [None] * 7 + [D] + [None] * 3  # absent run 4..10, drones at 3 and 11

    assert empty_sky_frames(boxes, margin=3) == [7]  # 4, 5, 6 are within 3 of frame 3; 8, 9, 10 of frame 11


def test_a_frame_exactly_the_margin_from_a_drone_is_not_empty_sky():
    boxes = [D] + [None] * 4 + [D]  # frames 1..4; frame 3 is 2 from the drone at 5

    assert empty_sky_frames(boxes, margin=2) == []
    assert empty_sky_frames(boxes, margin=1) == [2, 3]


def test_the_frames_just_outside_the_clip_count_as_possibly_holding_the_drone():
    # Clips are consecutive pieces of one recording, so the drone may be just past either end.
    boxes = [None] * 10

    assert empty_sky_frames(boxes, margin=3) == [3, 4, 5, 6]


def test_a_frame_with_the_drone_labeled_is_never_empty_sky():
    assert empty_sky_frames([D] * 20, margin=0) == []


def test_frames_are_saved_as_half_open_ranges():
    assert to_ranges([3, 4, 5, 9, 11, 12]) == [[3, 6], [9, 10], [11, 13]]
    assert to_ranges([]) == []


def test_a_split_lists_each_clips_empty_sky_frames_and_their_minutes():
    clips = [Clip(name="a", fps=20.0, boxes=[None] * 100),  # margin 2 s = 40 frames: 40..59
             Clip(name="b", fps=20.0, boxes=[D] * 100),
             Clip(name="c", fps=10.0, boxes=[None] * 100)]  # margin 20 frames at 10 fps: 20..79

    split = empty_sky_split(clips, seconds=2.0)

    assert split["clips"] == {"a": [[40, 60]], "b": [], "c": [[20, 80]]}
    assert split["frames"] == 80
    assert split["minutes"] == pytest.approx(20 / 20 / 60 + 60 / 10 / 60)


def test_the_command_writes_val_and_test_with_the_rule(tmp_path, monkeypatch):
    clips = {"x": Clip(name="x", fps=20.0, boxes=[None] * 100), "y": Clip(name="y", fps=20.0, boxes=[D] * 5)}
    monkeypatch.setattr("tether.empty_sky.load_clip", lambda d: clips[d.name])
    split = {"seed": 0, "frames": {"val/x": 100, "test/y": 5},
             "clips": {"train": [], "val": ["val/x"], "test": ["test/y"]}}
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split), encoding="utf-8")
    out = tmp_path / "sky.json"

    assert main(["data/Anti-UAV300", "--split-file", str(split_file), "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["margin_seconds"] == 2.0
    assert "rule" in saved
    assert saved["splits"]["val"]["clips"] == {"x": [[40, 60]]}
    assert saved["splits"]["test"]["clips"] == {"y": []}
    assert "train" not in saved["splits"]
    assert load_empty_sky(out, "val") == {"x": set(range(40, 60))}
    # How val alone sized the margin: frames kept at other margins, on val only.
    assert saved["val_sizing"] == {"0.5": 80, "1.0": 60, "2.0": 20, "5.0": 0}
