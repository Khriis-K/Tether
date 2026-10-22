"""Scoring a detector on Anti-UAV clips (#42): AP50, recall/precision/F1 at a threshold, recall by
box size, and false alarms per minute on drone-absent frames. Null cases first: they prove the
metric before any model is scored with it."""

import json

import pytest

from tether.antiuav import Clip
from tether.drone_score import (ap50, at_threshold, best_f1_threshold, main, match_clip, merge)
from tether.formats import Detection, save_detections


def drone(frame, size=40.0):
    """A drone that wanders over a 1920x1080 frame, so no fixed box can follow it."""
    x = 100.0 + (frame * 137) % 1700
    y = 100.0 + (frame * 89) % 900
    return (x, y, x + size, y + size)


def det(box, score=0.9, cls=0):
    return Detection(box=box, score=score, cls=cls)


def flying_clip(n=200, fps=20.0, name="c1"):
    return Clip(name=name, fps=fps, boxes=[drone(f) for f in range(n)])


# --- Null cases ---

def test_a_detector_that_outputs_nothing_scores_zero():
    clip = flying_clip()

    matches = match_clip(clip, [[] for _ in clip.boxes])

    assert ap50(matches) == 0.0
    assert at_threshold(matches, 0.25)["recall"] == 0.0


def test_a_detector_that_outputs_the_same_box_every_frame_scores_about_zero():
    clip = flying_clip()
    fixed = drone(0)

    matches = match_clip(clip, [[det(fixed)] for _ in clip.boxes])

    assert ap50(matches) < 0.02


def test_a_detector_that_copies_the_labeled_boxes_scores_one():
    clip = flying_clip()

    matches = match_clip(clip, [[det(b, score=1.0)] for b in clip.boxes])

    assert ap50(matches) == pytest.approx(1.0)
    scores = at_threshold(matches, 0.25)
    assert (scores["recall"], scores["precision"], scores["f1"]) == (1.0, 1.0, 1.0)


def test_a_detector_that_boxes_the_whole_frame_finds_nothing():
    clip = flying_clip()

    matches = match_clip(clip, [[det((0.0, 0.0, 1920.0, 1080.0))] for _ in clip.boxes])

    assert ap50(matches) == 0.0


# --- Matching ---

def test_a_detection_is_a_hit_from_iou_of_one_half():
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)] * 2)
    half = (0.0, 0.0, 10.0, 20.0)  # IoU 100 / 200 = 0.5
    just_under = (0.0, 0.0, 10.0, 20.5)

    assert at_threshold(match_clip(clip, [[det(half)], [det(half)]]), 0.25)["recall"] == 1.0
    assert at_threshold(match_clip(clip, [[det(just_under)], [det(just_under)]]), 0.25)["recall"] == 0.0


def test_a_second_box_on_the_drone_is_a_false_positive():
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)])

    scores = at_threshold(match_clip(clip, [[det((0.0, 0.0, 10.0, 10.0)), det((0.0, 0.0, 10.0, 11.0), 0.8)]]), 0.25)

    assert (scores["found"], scores["detections"], scores["precision"]) == (1, 2, 0.5)


def test_the_drone_is_found_by_its_best_scored_overlapping_detection():
    # A low-scored hit and a high-scored miss: above the low score, the drone isn't found.
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)])
    matches = match_clip(clip, [[det((0.0, 0.0, 10.0, 10.0), 0.3), det((50.0, 50.0, 60.0, 60.0), 0.9)]])

    assert at_threshold(matches, 0.5)["recall"] == 0.0
    assert at_threshold(matches, 0.25)["recall"] == 1.0


def test_classes_are_ignored():
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)])

    assert at_threshold(match_clip(clip, [[det((0.0, 0.0, 10.0, 10.0), cls=14)]]), 0.25)["recall"] == 1.0


def test_detections_below_the_threshold_are_not_counted():
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0), None])
    matches = match_clip(clip, [[det((0.0, 0.0, 10.0, 10.0), 0.2)], [det((5.0, 5.0, 9.0, 9.0), 0.2)]])

    scores = at_threshold(matches, 0.25)

    assert (scores["found"], scores["detections"], scores["false_alarms"]) == (0, 0, 0)
    assert scores["precision"] is None  # nothing reported, so no precision rather than a flattering 0/0


def test_frames_the_tool_lacks_detections_for_are_refused():
    clip = flying_clip(n=5)

    with pytest.raises(ValueError):
        match_clip(clip, [[] for _ in range(4)])


# --- AP50 ---

def test_ap50_is_the_area_under_the_precision_envelope():
    # Four drones. By score: hit, miss, hit, miss. Curve points (recall, precision):
    # (0.25, 1), (0.25, 0.5), (0.5, 0.67), (0.5, 0.5). Envelope area: 0.25 * 1 + 0.25 * 2/3.
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 4
    away = (50.0, 50.0, 60.0, 60.0)
    frames = [[det(boxes[0], 0.9)], [det(away, 0.8)], [det(boxes[2], 0.7)], [det(away, 0.6)]]

    assert ap50(match_clip(Clip(name="c", fps=20.0, boxes=boxes), frames)) == pytest.approx(0.25 + 0.25 * 2 / 3)


def test_ap50_does_not_depend_on_the_order_of_tied_scores():
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 2
    away = (50.0, 50.0, 60.0, 60.0)
    hit_first = match_clip(Clip(name="c", fps=20.0, boxes=boxes), [[det(boxes[0], 0.5)], [det(away, 0.5)]])
    miss_first = match_clip(Clip(name="c", fps=20.0, boxes=boxes), [[det(away, 0.5)], [det(boxes[1], 0.5)]])

    assert ap50(hit_first) == ap50(miss_first) == pytest.approx(0.25)


def test_ap50_with_no_drone_labeled_is_none():
    clip = Clip(name="c", fps=20.0, boxes=[None, None])

    assert ap50(match_clip(clip, [[det((0.0, 0.0, 9.0, 9.0))], []])) is None


# --- The operating threshold ---

def test_the_best_f1_threshold_is_the_score_cut_with_the_highest_f1():
    # Hits at 0.9 and 0.6, misses at 0.7 and 0.3: cutting at 0.6 finds both drones with one false alarm.
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 3
    away = (50.0, 50.0, 60.0, 60.0)
    frames = [[det(boxes[0], 0.9), det(away, 0.7)], [det(boxes[1], 0.6)], [det(away, 0.3)]]

    matches = match_clip(Clip(name="c", fps=20.0, boxes=boxes), frames)

    assert best_f1_threshold(matches) == pytest.approx(0.6)


def test_with_no_detections_there_is_no_best_threshold():
    assert best_f1_threshold(match_clip(flying_clip(n=3), [[], [], []])) is None


# --- Recall by box size ---

def test_recall_is_split_by_coco_style_box_size_at_source_resolution():
    sizes = [10.0, 20.0, 50.0, 120.0]  # tiny < 16 px, small < 32, medium < 96, large: square side
    boxes = [(0.0, 0.0, s, s) for s in sizes] * 2
    frames = [[det(b)] for b in boxes[:4]] + [[] for _ in range(4)]  # each size found once in two

    by_size = at_threshold(match_clip(Clip(name="c", fps=20.0, boxes=boxes), frames), 0.25)["recall_by_size"]

    assert by_size == {name: {"found": 1, "drones": 2, "recall": 0.5}
                       for name in ("tiny", "small", "medium", "large")}


# --- False alarms per minute ---

def test_false_alarms_count_only_frames_where_the_drone_is_labeled_absent():
    boxes = [(0.0, 0.0, 10.0, 10.0), None, None]
    away = (50.0, 50.0, 60.0, 60.0)
    frames = [[det(away)], [det(away), det((0.0, 0.0, 10.0, 10.0))], []]

    scores = at_threshold(match_clip(Clip(name="c", fps=20.0, boxes=boxes), frames), 0.25)

    assert scores["false_alarms"] == 2
    assert scores["absent_minutes"] == pytest.approx(2 / 20 / 60)
    assert scores["false_alarms_per_minute"] == pytest.approx(2 / (2 / 20 / 60))


def test_false_alarms_per_minute_use_each_clips_own_frame_rate():
    # One false alarm on each of 600 absent frames: 30 s at 20 fps, 60 s at 10 fps.
    fast = Clip(name="a", fps=20.0, boxes=[None] * 600)
    slow = Clip(name="b", fps=10.0, boxes=[None] * 600)
    one_each = [[det((0.0, 0.0, 9.0, 9.0))] for _ in range(600)]

    scores = at_threshold(merge([match_clip(fast, one_each), match_clip(slow, one_each)]), 0.25)

    assert scores["absent_minutes"] == pytest.approx(1.5)
    assert scores["false_alarms_per_minute"] == pytest.approx(1200 / 1.5)


def test_with_no_drone_absent_frames_there_is_no_false_alarm_rate():
    scores = at_threshold(match_clip(flying_clip(n=3), [[], [], []]), 0.25)

    assert scores["false_alarms_per_minute"] is None


def test_merging_clips_pools_drones_and_detections():
    a = Clip(name="a", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)])
    b = Clip(name="b", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)])

    merged = merge([match_clip(a, [[det(a.boxes[0])]]), match_clip(b, [[]])])

    assert at_threshold(merged, 0.25)["recall"] == 0.5


# --- The command ---

def write_split(tmp_path, clip_names):
    split = {"seed": 0, "clips": {"train": [], "val": [f"val/{c}" for c in clip_names], "test": []},
             "frames": {f"val/{c}": 1 for c in clip_names}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    return path


def test_the_command_saves_results_naming_model_split_threshold_and_iou_rule(tmp_path, monkeypatch):
    clips = {"c1": flying_clip(n=50, name="c1"), "c2": Clip(name="c2", fps=10.0, boxes=[None] * 50)}
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clips[d.name])
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[det(b, 0.8)] for b in clips["c1"].boxes])
    save_detections(dets / "c2.jsonl", [[] for _ in range(50)])
    (dets / "dump.json").write_text(json.dumps({"model": "yolo26n_1280x736", "min_score": 0.1}), encoding="utf-8")
    out = tmp_path / "results.json"

    assert main(["data/Anti-UAV300", str(dets), "--split-file", str(write_split(tmp_path, ["c1", "c2"])),
                 "--out", str(out)]) == 0

    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["model"] == "yolo26n_1280x736"
    assert results["split"] == {"file": str(tmp_path / "split.json"), "name": "val", "seed": 0}
    assert results["min_iou"] == 0.5
    assert results["min_score"] == 0.1
    assert results["ap50"] == pytest.approx(1.0)
    assert [t["threshold"] for t in results["thresholds"]] == [pytest.approx(0.8), 0.25]
    assert results["thresholds"][0]["chosen_by"] == "best F1 on this split"
    assert results["thresholds"][0]["recall"] == 1.0


def test_a_given_threshold_replaces_the_best_f1_one(tmp_path, monkeypatch):
    clip = flying_clip(n=10, name="c1")
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clip)
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[det(b, 0.8)] for b in clip.boxes])
    out = tmp_path / "results.json"

    main(["data/Anti-UAV300", str(dets), "--split-file", str(write_split(tmp_path, ["c1"])),
          "--threshold", "0.5", "--out", str(out)])

    results = json.loads(out.read_text(encoding="utf-8"))
    assert [t["threshold"] for t in results["thresholds"]] == [0.5, 0.25]
    assert results["thresholds"][0]["chosen_by"] == "given"
    assert results["model"] == "dets"  # no dump.json: named after the detection folder


def test_a_threshold_below_the_dumps_score_floor_is_refused(tmp_path, monkeypatch):
    # Detections under the floor were never saved, so recall there would be understated.
    clip = flying_clip(n=10, name="c1")
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clip)
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[det(b, 0.8)] for b in clip.boxes])
    (dets / "dump.json").write_text(json.dumps({"model": "m", "min_score": 0.3}), encoding="utf-8")

    with pytest.raises(ValueError):  # the tracker's 0.25 row is under 0.3
        main(["data/Anti-UAV300", str(dets), "--split-file", str(write_split(tmp_path, ["c1"])),
              "--threshold", "0.5", "--out", str(tmp_path / "results.json")])

    assert not (tmp_path / "results.json").exists()


def test_a_best_f1_cut_at_the_lowest_saved_score_is_flagged(tmp_path, monkeypatch):
    # F1 may still be rising below the lowest score the dump kept, so that cut isn't a real optimum.
    clips = {"c1": Clip(name="c1", fps=20.0, boxes=[DRONE_BOX] * 2), "c2": Clip(name="c2", fps=20.0, boxes=[DRONE_BOX])}
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clips[d.name])
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[det(DRONE_BOX, 0.9)], [det(DRONE_BOX, 0.3)]])  # best cut 0.3, the lowest
    save_detections(dets / "c2.jsonl", [[det((50.0, 50.0, 60.0, 60.0), 0.5)]])
    out = tmp_path / "results.json"

    main(["data/Anti-UAV300", str(dets), "--split-file", str(write_split(tmp_path, ["c1", "c2"])), "--out", str(out)])

    rows = json.loads(out.read_text(encoding="utf-8"))["thresholds"]
    assert rows[0]["threshold"] == pytest.approx(0.3)
    assert rows[0]["at_lowest_saved_score"] is True
    assert rows[1]["at_lowest_saved_score"] is False


DRONE_BOX = (0.0, 0.0, 10.0, 10.0)
