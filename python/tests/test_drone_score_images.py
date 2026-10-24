"""Scoring a detector on folders of images (#45): Seraphim test (YOLO-format labels, any number
of drones per image) for the transfer test, and drone-free images (Drone-vs-Bird birds) for the
per-image false-alarm rate. Also the empty-sky restriction of the Anti-UAV false-alarm row.
Null cases first, the same ones as the clip scorer's (#42)."""

import json

import pytest

from tether.antiuav import Clip
from tether.drone_score import (ap50, at_threshold, drone_free_rate, evaluate, load_yolo_labels, main_images,
                                match_clip, match_images)
from tether.formats import Detection, save_detections


def det(box, score=0.9, cls=0):
    return Detection(box=box, score=score, cls=cls)


def drone(i, size=40.0):
    """A drone that wanders over a 640x640 image, so no fixed box can follow it."""
    x = 20.0 + (i * 137) % 560
    y = 20.0 + (i * 89) % 560
    return (x, y, x + size, y + size)


def image_set(n=100):
    """n images; every fifth holds a second drone."""
    return [[drone(i)] + ([drone(i + 1000, 30.0)] if i % 5 == 0 else []) for i in range(n)]


# --- Null cases ---

def test_a_detector_that_outputs_nothing_scores_zero():
    truth = image_set()

    matches = match_images(truth, [[] for _ in truth])

    assert ap50(matches) == 0.0
    assert at_threshold(matches, 0.25)["recall"] == 0.0


def test_a_detector_that_outputs_the_same_box_on_every_image_scores_about_zero():
    truth = image_set()

    assert ap50(match_images(truth, [[det(drone(0))] for _ in truth])) < 0.02


def test_a_detector_that_copies_the_labels_scores_one():
    truth = image_set()

    matches = match_images(truth, [[det(b, 1.0) for b in boxes] for boxes in truth])

    assert ap50(matches) == pytest.approx(1.0)
    scores = at_threshold(matches, 0.25)
    assert (scores["recall"], scores["precision"], scores["f1"]) == (1.0, 1.0, 1.0)
    assert scores["drones"] == 120


def test_a_detector_that_boxes_the_whole_image_finds_nothing():
    truth = image_set()

    assert ap50(match_images(truth, [[det((0.0, 0.0, 640.0, 640.0))] for _ in truth])) == 0.0


# --- Matching several drones per image ---

def test_each_drone_in_an_image_can_be_found_by_its_own_detection():
    a, b = (0.0, 0.0, 10.0, 10.0), (100.0, 100.0, 110.0, 110.0)

    scores = at_threshold(match_images([[a, b]], [[det(a), det(b, 0.5)]]), 0.25)

    assert (scores["found"], scores["drones"], scores["detections"]) == (2, 2, 2)


def test_one_detection_finds_at_most_one_drone():
    a, b = (0.0, 0.0, 10.0, 10.0), (1.0, 0.0, 11.0, 10.0)  # two drones overlapping at IoU 0.82

    scores = at_threshold(match_images([[a, b]], [[det(a)]]), 0.25)

    assert (scores["found"], scores["drones"]) == (1, 2)


def test_detections_claim_drones_in_score_order_and_a_second_box_on_a_drone_is_a_false_positive():
    a = (0.0, 0.0, 10.0, 10.0)
    matches = match_images([[a]], [[det((0.0, 0.0, 10.0, 11.0), 0.4), det(a, 0.8)]])

    at_high, at_low = at_threshold(matches, 0.5), at_threshold(matches, 0.25)

    assert (at_high["found"], at_high["detections"]) == (1, 1)  # the 0.8 box found it
    assert (at_low["found"], at_low["detections"], at_low["precision"]) == (1, 2, 0.5)


def test_a_detection_claims_the_drone_it_overlaps_most():
    # The 0.9 box overlaps a at IoU 0.69 and b at 0.85; it takes b, leaving a for the 0.5 box, which
    # overlaps a at 0.67 and b at only 0.38. Taking the first drone over 0.5 instead would find one.
    a, b = (0.0, 0.0, 10.0, 10.0), (0.0, 2.0, 10.0, 14.0)
    first, second = (0.0, 1.0, 10.0, 13.0), (0.0, -2.0, 10.0, 8.0)

    scores = at_threshold(match_images([[a, b]], [[det(first, 0.9), det(second, 0.5)]]), 0.25)

    assert scores["found"] == 2


def test_an_image_without_detections_for_it_is_refused():
    with pytest.raises(ValueError):
        match_images(image_set(5), [[] for _ in range(4)])


# --- YOLO-format labels ---

def test_yolo_labels_are_turned_into_pixel_corners(tmp_path):
    label = tmp_path / "1.txt"
    label.write_text("0 0.5 0.5 0.25 0.5\n0 0.1 0.2 0.1 0.2\n\n", encoding="utf-8")

    boxes = load_yolo_labels(label, width=640, height=480)

    assert boxes == [pytest.approx((240.0, 120.0, 400.0, 360.0)), pytest.approx((32.0, 48.0, 96.0, 144.0))]


def test_an_empty_label_file_means_no_drone(tmp_path):
    label = tmp_path / "1.txt"
    label.write_text("", encoding="utf-8")

    assert load_yolo_labels(label, width=640, height=640) == []


# --- The per-image false-alarm rate on drone-free images ---

def test_no_detections_is_a_false_alarm_rate_of_zero():
    rate = drone_free_rate([[] for _ in range(50)], 0.25)

    assert (rate["images"], rate["with_alarm"], rate["rate"]) == (50, 0, 0.0)


def test_a_box_on_every_image_is_a_false_alarm_rate_of_one():
    assert drone_free_rate([[det(drone(i), 0.3)] for i in range(50)], 0.25)["rate"] == 1.0


def test_an_image_counts_once_however_many_boxes_it_has():
    frames = [[det(drone(0)), det(drone(1)), det(drone(2))], [], [], []]

    assert drone_free_rate(frames, 0.25)["with_alarm"] == 1
    assert drone_free_rate(frames, 0.25)["rate"] == 0.25


def test_boxes_below_the_threshold_raise_no_alarm():
    frames = [[det(drone(0), 0.2)], [det(drone(1), 0.25)]]

    assert drone_free_rate(frames, 0.25)["with_alarm"] == 1


def test_with_no_images_there_is_no_rate():
    assert drone_free_rate([], 0.25)["rate"] is None


# --- The image-folder command ---

def write_dump(tmp_path, frames, names, sizes, model="yolo26n_1280x736", min_score=0.01):
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "detections.jsonl", frames)
    (dets / "dump.json").write_text(json.dumps({"model": model, "min_score": min_score, "set": "seraphim-test",
                                                "images": names, "sizes": sizes}), encoding="utf-8")
    return dets


def test_the_command_scores_labeled_images_naming_model_set_and_threshold(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "7.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")  # 100x200 image: (25, 50, 75, 150)
    (labels / "8.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    dets = write_dump(tmp_path, [[det((25.0, 50.0, 75.0, 150.0), 0.8)], [det((0.0, 0.0, 5.0, 5.0), 0.6)]],
                      ["7.jpg", "8.jpg"], [[100, 200], [100, 200]])
    out = tmp_path / "results.json"

    assert main_images([str(dets), "--labels", str(labels), "--threshold", "0.066", "--out", str(out)]) == 0

    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["model"] == "yolo26n_1280x736"
    assert results["set"] == "seraphim-test"
    assert results["images"] == 2
    assert results["min_iou"] == 0.5
    assert results["ap50"] == pytest.approx(0.5)
    assert [t["threshold"] for t in results["thresholds"]] == [0.066, 0.25]
    assert results["thresholds"][0]["chosen_by"] == "given"
    assert (results["thresholds"][0]["found"], results["thresholds"][0]["drones"]) == (1, 2)


def test_the_command_gives_drone_free_images_a_per_image_false_alarm_rate(tmp_path):
    dets = write_dump(tmp_path, [[det((0.0, 0.0, 5.0, 5.0), 0.3)], [det((0.0, 0.0, 5.0, 5.0), 0.1)], [], []],
                      ["BT (1).jpg", "BT (2).jpg", "BT (3).jpg", "BT (4).jpg"], [[640, 640]] * 4)
    out = tmp_path / "results.json"

    assert main_images([str(dets), "--drone-free", "--threshold", "0.066", "--out", str(out)]) == 0

    rows = json.loads(out.read_text(encoding="utf-8"))["thresholds"]
    assert [(r["threshold"], r["with_alarm"], r["images"], r["rate"]) for r in rows] == [
        (0.066, 2, 4, 0.5), (0.25, 1, 4, 0.25)]


def test_the_command_refuses_a_threshold_below_the_dumps_floor(tmp_path):
    dets = write_dump(tmp_path, [[]], ["BT (1).jpg"], [[640, 640]], min_score=0.1)

    with pytest.raises(ValueError):
        main_images([str(dets), "--drone-free", "--threshold", "0.066", "--out", str(tmp_path / "r.json")])


def test_the_command_refuses_a_dump_whose_length_differs_from_its_image_list(tmp_path):
    dets = write_dump(tmp_path, [[], []], ["BT (1).jpg"], [[640, 640]])

    with pytest.raises(ValueError):
        main_images([str(dets), "--drone-free", "--out", str(tmp_path / "r.json")])


def test_the_command_needs_labels_or_drone_free(tmp_path):
    dets = write_dump(tmp_path, [[]], ["BT (1).jpg"], [[640, 640]])

    with pytest.raises(SystemExit):
        main_images([str(dets), "--out", str(tmp_path / "r.json")])


# --- Empty sky: false alarms on listed drone-absent frames only ---

def test_with_empty_sky_frames_given_only_those_frames_count_as_false_alarms():
    away = (50.0, 50.0, 60.0, 60.0)
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0), None, None, None])
    frames = [[], [det(away)], [det(away), det(away)], [det(away)]]

    scores = at_threshold(match_clip(clip, frames, sky={2, 3}), 0.25)

    assert scores["false_alarms"] == 3
    assert scores["absent_minutes"] == pytest.approx(2 / 20 / 60)


def test_an_empty_sky_frame_with_the_drone_labeled_is_refused():
    clip = Clip(name="c", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0), None])

    with pytest.raises(ValueError):
        match_clip(clip, [[], []], sky={0})


def test_the_clip_command_scores_false_alarms_on_the_empty_sky_list(tmp_path, monkeypatch):
    clip = Clip(name="c1", fps=20.0, boxes=[(0.0, 0.0, 10.0, 10.0)] + [None] * 99)
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clip)
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[det((50.0, 50.0, 60.0, 60.0), 0.9)] for _ in range(100)])
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"seed": 0, "frames": {"val/c1": 100},
                                      "clips": {"train": [], "val": ["val/c1"], "test": []}}), encoding="utf-8")
    sky = tmp_path / "sky.json"
    sky.write_text(json.dumps({"splits": {"val": {"clips": {"c1": [[60, 70]]}}}}), encoding="utf-8")

    results = evaluate("data/Anti-UAV300", split_file, "val", dets, threshold=0.5, empty_sky=sky)

    assert results["empty_sky"] == str(sky)
    row = results["thresholds"][0]
    assert (row["false_alarms"], row["absent_minutes"]) == (10, pytest.approx(10 / 20 / 60))


def test_a_clip_missing_from_the_empty_sky_list_is_refused(tmp_path, monkeypatch):
    # A list built for another split would otherwise score 0 false alarms in 0 minutes, silently.
    clip = Clip(name="c1", fps=20.0, boxes=[None] * 10)
    monkeypatch.setattr("tether.drone_score.load_clip", lambda d: clip)
    dets = tmp_path / "dets"
    dets.mkdir()
    save_detections(dets / "c1.jsonl", [[] for _ in range(10)])
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"seed": 0, "frames": {"val/c1": 10},
                                      "clips": {"train": [], "val": ["val/c1"], "test": []}}), encoding="utf-8")
    sky = tmp_path / "sky.json"
    sky.write_text(json.dumps({"splits": {"val": {"clips": {"other": []}}}}), encoding="utf-8")

    with pytest.raises(ValueError) as refused:
        evaluate("data/Anti-UAV300", split_file, "val", dets, threshold=0.5, empty_sky=sky)

    assert "val" in str(refused.value)
