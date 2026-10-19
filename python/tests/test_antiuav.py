import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from tether.antiuav import (DRONE, Clip, blank, drop_random, evaluate, gap_recovery, id_kept, load_clip,
                            place_gaps, seconds_to_frames, to_detections)
from tether.formats import Track, load_detections


def drone_box(frame, speed=10.0, size=40.0):
    """A drone flying right at `speed` px per frame."""
    x = 100.0 + speed * frame
    return (x, 200.0, x + size, 200.0 + size)


def track(id, box, status="tracked", predicted_box=None):
    return Track(id=id, box=box, cls=DRONE, status=status, predicted_box=predicted_box or box,
                 frames_since_seen=0 if status != "searching" else 1, path=())


def write_clip(split_dir, name, boxes, fps=20.0):
    """An Anti-UAV300 clip folder: visible.mp4 plus visible.json ({exist, gt_rect as x, y, w, h})."""
    clip = split_dir / name
    clip.mkdir(parents=True)
    exist = [int(b is not None) for b in boxes]
    rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] if b is not None else [] for b in boxes]
    (clip / "visible.json").write_text(json.dumps({"exist": exist, "gt_rect": rects}), encoding="utf-8")
    video = cv2.VideoWriter(str(clip / "visible.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in boxes:
        video.write(np.zeros((48, 64, 3), np.uint8))
    video.release()
    return clip


# --- Loading a clip ---

def test_loads_boxes_as_corners_and_reads_the_frame_rate_from_the_video(tmp_path):
    clip_dir = write_clip(tmp_path, "c1", [(10.0, 20.0, 50.0, 60.0), (12.0, 20.0, 52.0, 60.0)], fps=25.0)

    clip = load_clip(clip_dir)

    assert clip.name == "c1"
    assert clip.fps == pytest.approx(25.0)
    assert clip.boxes == [(10.0, 20.0, 50.0, 60.0), (12.0, 20.0, 52.0, 60.0)]


def test_frames_marked_not_existing_have_no_drone_whatever_their_rect_says(tmp_path):
    # The real labels mark an absent drone with [], [0, 0, 0, 0], or now and then a leftover box.
    clip_dir = tmp_path / "c1"
    write_clip(tmp_path, "c1", [None, None, None, (1.0, 2.0, 4.0, 6.0)])
    (clip_dir / "visible.json").write_text(json.dumps(
        {"exist": [0, 0, 0, 1], "gt_rect": [[], [0, 0, 0, 0], [5, 5, 10, 10], [1, 2, 3, 4]]}), encoding="utf-8")

    assert load_clip(clip_dir).boxes == [None, None, None, (1.0, 2.0, 4.0, 6.0)]


def test_a_clip_whose_labels_and_video_disagree_on_length_is_refused(tmp_path):
    clip_dir = write_clip(tmp_path, "c1", [drone_box(f) for f in range(5)])
    (clip_dir / "visible.json").write_text(json.dumps(
        {"exist": [1] * 4, "gt_rect": [[0, 0, 1, 1]] * 4}), encoding="utf-8")

    with pytest.raises(ValueError, match="4 labels"):
        load_clip(clip_dir)


def test_labeled_boxes_become_one_full_confidence_drone_detection_per_frame():
    frames = to_detections([(1.0, 2.0, 3.0, 4.0), None])

    assert len(frames) == 2 and frames[1] == []
    (det,) = frames[0]
    assert (det.box, det.score, det.cls) == ((1.0, 2.0, 3.0, 4.0), 1.0, DRONE)


# --- Random deletion ---

BOXES = [drone_box(f) for f in range(10_000)]


def test_random_deletion_removes_about_the_given_share_of_boxes():
    kept = drop_random(BOXES, 0.2, seed=0)

    assert len(kept) == len(BOXES)
    assert sum(b is None for b in kept) / len(BOXES) == pytest.approx(0.2, abs=0.02)
    assert all(k is None or k == b for k, b in zip(kept, BOXES))


def test_random_deletion_is_repeatable_with_the_same_seed_and_differs_with_another():
    assert drop_random(BOXES, 0.2, seed=7) == drop_random(BOXES, 0.2, seed=7)
    assert drop_random(BOXES, 0.2, seed=7) != drop_random(BOXES, 0.2, seed=8)


def test_deletion_rates_zero_and_one_keep_all_and_none():
    assert drop_random(BOXES, 0.0, seed=0) == BOXES
    assert drop_random(BOXES, 1.0, seed=0) == [None] * len(BOXES)


def test_frames_with_no_drone_stay_empty():
    assert drop_random([None, None], 0.0, seed=0) == [None, None]


# --- Gaps ---

def test_gap_seconds_become_frames_at_the_clips_frame_rate():
    assert seconds_to_frames(1.0, 20.0) == 20
    assert seconds_to_frames(0.5, 30.0) == 15
    assert seconds_to_frames(0.25, 25.0) == 6  # 6.25 rounds to 6
    assert seconds_to_frames(0.01, 20.0) == 1  # a gap is never empty


def test_gaps_follow_settle_stretches_and_each_leaves_frames_after_it():
    # settle 5, gap 3, repeating: [5, 8), [13, 16); a third at [21, 24) would leave nothing after it.
    assert place_gaps(24, gap_frames=3, settle_frames=5) == [(5, 8), (13, 16)]
    assert place_gaps(25, gap_frames=3, settle_frames=5) == [(5, 8), (13, 16), (21, 24)]
    assert place_gaps(8, gap_frames=3, settle_frames=5) == []


def test_blanking_removes_boxes_inside_gaps_only():
    boxes = [drone_box(f) for f in range(6)]

    assert blank(boxes, [(1, 3), (4, 5)]) == [boxes[0], None, None, boxes[3], None, boxes[5]]


# --- ID kept ---

GT = [drone_box(f) for f in range(10)]


def test_a_perfect_tracker_keeps_the_id_on_every_frame():
    assert id_kept(GT, GT, [[track(1, b)] for b in GT]) == {"kept": 10, "frames": 10}


def test_frames_whose_detection_was_deleted_are_not_scored():
    given = [b if f % 2 == 0 else None for f, b in enumerate(GT)]
    frames = [[track(1, b, status="tracked" if f % 2 == 0 else "searching")] for f, b in enumerate(GT)]

    assert id_kept(GT, given, frames) == {"kept": 5, "frames": 5}


def test_the_id_counts_only_where_the_track_overlaps_the_drone_with_iou_of_at_least_half():
    # Shifted 15 px on a 40 px box: IoU 25/55 < 0.5. Shifted 10 px: IoU 30/50 = 0.6.
    frames = [[track(1, drone_box(f + (1.5 if f < 4 else 1.0)))] for f in range(10)]

    assert id_kept(GT, GT, frames) == {"kept": 6, "frames": 10}


def test_an_iou_of_exactly_half_holds_the_drone():
    # Twice as wide and containing the drone: IoU 1600 / 3200.
    frames = [[track(1, (b[0], b[1], b[2] + 40.0, b[3]))] for b in GT]

    assert id_kept(GT, GT, frames)["kept"] == 10


def test_a_track_that_covers_the_whole_image_never_keeps_the_id():
    # The track that never dies: it always "contains" the drone, but doesn't locate it.
    frames = [[track(1, (0.0, 0.0, 1920.0, 1080.0))] for _ in GT]

    assert id_kept(GT, GT, frames)["kept"] == 0


def test_tentative_and_searching_tracks_do_not_hold_the_id():
    frames = [[track(1, b, status="tentative" if f < 3 else "searching" if f < 5 else "tracked")]
              for f, b in enumerate(GT)]

    assert id_kept(GT, GT, frames) == {"kept": 5, "frames": 10}


def test_after_an_id_switch_only_the_most_common_id_counts():
    frames = [[track(1 if f < 3 else 2, b)] for f, b in enumerate(GT)]

    assert id_kept(GT, GT, frames) == {"kept": 7, "frames": 10}


def test_a_tracker_with_no_tracks_keeps_nothing():
    assert id_kept(GT, GT, [[] for _ in GT]) == {"kept": 0, "frames": 10}


# --- Recovery across gaps ---

N = 30
CLIP = [drone_box(f) for f in range(N)]
GAPS = [(10, 15)]


def follower(gt, gaps, *, new_id_after_gap=False, predict=lambda f: None):
    """A tracker that is exactly on the drone outside gaps and searching inside them.

    `predict(f)` gives the predicted box it reports on frame f (for frame f + 1);
    by default it predicts the drone exactly.
    """
    in_gap = {f for s, e in gaps for f in range(s, e)}
    frames = []
    for f, box in enumerate(gt):
        after = new_id_after_gap and f >= gaps[0][1]
        last_seen = max(g for g in range(f + 1) if g not in in_gap)
        pred = predict(f) or (gt[f + 1] if f + 1 < len(gt) else box)
        frames.append([track(2 if after else 1, gt[last_seen],
                             status="searching" if f in in_gap else "tracked", predicted_box=pred)])
    return frames


def test_the_same_id_back_on_the_drone_after_a_gap_is_recovered():
    result = gap_recovery(CLIP, GAPS, follower(CLIP, GAPS))

    assert (result["gaps"], result["held_before"], result["recovered"]) == (1, 1, 1)


def test_a_new_id_after_the_gap_is_not_recovered():
    result = gap_recovery(CLIP, GAPS, follower(CLIP, GAPS, new_id_after_gap=True))

    assert (result["held_before"], result["recovered"]) == (1, 0)


def test_a_tracker_that_freezes_its_box_is_never_recovered():
    # Exactly on the drone up to the gap, then it keeps ID 1 and "tracked" forever but stops moving.
    # The drone moves 10 px a frame on a 40 px box, so the frozen box falls below IoU 0.5 within 2 frames.
    s, _ = GAPS[0]
    frames = [[track(1, CLIP[min(f, s - 1)])] for f in range(N)]

    result = gap_recovery(CLIP, GAPS, frames)

    assert result["held_before"] == 1
    assert result["recovered"] == 0


def test_a_gap_the_tracker_was_not_holding_before_is_counted_but_not_held():
    frames = follower(CLIP, GAPS)
    frames[9] = []

    result = gap_recovery(CLIP, GAPS, frames)

    assert (result["gaps"], result["held_before"], result["recovered"]) == (1, 0, 0)


def test_recovery_is_checked_on_the_first_frame_the_drone_is_labeled_after_the_gap():
    gt = CLIP[:15] + [None, None] + CLIP[17:]
    frames = follower(CLIP, GAPS)
    frames[15] = frames[16] = []  # no drone labeled there, so these frames don't matter

    assert gap_recovery(gt, GAPS, frames)["recovered"] == 1

    frames[17] = []
    assert gap_recovery(gt, GAPS, frames)["recovered"] == 0


def test_gaps_with_no_drone_labeled_just_before_or_anywhere_after_are_not_scored():
    no_before = CLIP[:9] + [None] + CLIP[10:]
    no_after = CLIP[:15] + [None] * (N - 15)

    assert gap_recovery(no_before, GAPS, follower(CLIP, GAPS))["gaps"] == 0
    assert gap_recovery(no_after, GAPS, follower(CLIP, GAPS))["gaps"] == 0


def test_a_perfect_prediction_has_zero_error_on_every_gap_frame():
    result = gap_recovery(CLIP, GAPS, follower(CLIP, GAPS))

    assert result["center_errors"] == [0.0] * 5
    assert result["ious"] == pytest.approx([1.0] * 5)


def test_prediction_error_is_the_distance_from_the_previous_frames_prediction_to_the_drone():
    # Predicts the drone 3 px right and 4 px down of where it will be: 5 px off.
    def predict(f):
        if f + 1 == N:
            return None
        x1, y1, x2, y2 = CLIP[f + 1]
        return (x1 + 3, y1 + 4, x2 + 3, y2 + 4)

    result = gap_recovery(CLIP, GAPS, follower(CLIP, GAPS, predict=predict))

    assert result["center_errors"] == pytest.approx([5.0] * 5)
    assert result["ious"] == pytest.approx([(37 * 36) / (2 * 1600 - 37 * 36)] * 5)


def test_no_prediction_error_is_measured_for_a_gap_that_was_not_held():
    frames = follower(CLIP, GAPS)
    frames[9] = []

    assert gap_recovery(CLIP, GAPS, frames)["center_errors"] == []


# --- The whole evaluation ---

TETHER = f'"{sys.executable}" -m tether.cli'


def steady_drone(n, speed=4.0):
    return [(600.0 + speed * f, 300.0, 640.0 + speed * f, 340.0) for f in range(n)]


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "Anti-UAV300"
    write_clip(root / "test", "clip-a", steady_drone(120))
    # Flies left, and isn't labeled for 10 frames after the gap's recovery frame.
    flies_left = steady_drone(100, speed=-3.0)
    write_clip(root / "test", "clip-b", flies_left[:80] + [None] * 10 + flies_left[90:])
    write_clip(root / "train", "clip-c", steady_drone(50))
    return root


def test_evaluates_the_split_and_saves_the_results(tmp_path, dataset):
    work = tmp_path / "work"

    results = evaluate(dataset, work, TETHER, "tether", split="test", drop_rates=[0.2], gap_seconds=[1.0])

    assert results["tracker"] == "tether"
    assert results["split"] == "test" and results["clips"] == 2
    (drop,) = results["random_deletion"]
    assert drop["drop_rate"] == 0.2
    assert drop["frames"] > 0 and 0 <= drop["kept"] <= drop["frames"]
    assert drop["id_kept"] == pytest.approx(100 * drop["kept"] / drop["frames"])
    (gap,) = results["gaps"]
    assert gap["gap_seconds"] == 1.0
    saved = json.loads((work / "trackers" / "tether" / "results.json").read_text(encoding="utf-8"))
    assert saved == results


def test_tether_recovers_a_steadily_moving_drone_across_one_second_gaps(tmp_path, dataset):
    results = evaluate(dataset, tmp_path / "work", TETHER, "tether", split="test",
                       drop_rates=[0.0], gap_seconds=[1.0])

    (gap,) = results["gaps"]
    assert gap["held_before"] >= 1
    assert gap["recovery"] == pytest.approx(100.0)
    assert gap["prediction_frames"] > 0
    assert gap["prediction_error_px"] >= 0.0


def test_runs_with_the_same_seed_give_the_same_results(tmp_path, dataset):
    first = evaluate(dataset, tmp_path / "a", TETHER, "tether", drop_rates=[0.3], gap_seconds=[0.5], seed=3)
    again = evaluate(dataset, tmp_path / "b", TETHER, "tether", drop_rates=[0.3], gap_seconds=[0.5], seed=3)

    assert first == again


def test_separate_runs_of_the_tool_delete_the_same_boxes(tmp_path, dataset):
    # Separate processes, with Python's per-process string hashing set differently.
    for run, hash_seed in (("a", "1"), ("b", "2")):
        subprocess.run([sys.executable, "-m", "tether.antiuav", str(dataset), "--work-dir", str(tmp_path / run),
                        "--drop-rates", "0.3", "--gap-seconds", "--seed", "5"],
                       check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})

    for clip in ("clip-a", "clip-b"):
        first, again = (tmp_path / run / "detections" / "drop-0.3-seed5" / f"{clip}.jsonl" for run in "ab")
        assert first.read_bytes() == again.read_bytes()
    assert (json.loads((tmp_path / "a" / "trackers" / "tether" / "results.json").read_text(encoding="utf-8"))
            == json.loads((tmp_path / "b" / "trackers" / "tether" / "results.json").read_text(encoding="utf-8")))


def test_gaps_are_blanked_using_each_clips_own_frame_rate(tmp_path):
    root = tmp_path / "Anti-UAV300"
    write_clip(root / "test", "slow", steady_drone(80), fps=10.0)
    work = tmp_path / "work"

    evaluate(root, work, TETHER, "tether", drop_rates=[], gap_seconds=[1.0])

    frames = load_detections(next((work / "detections").glob("gap-1.0s*/slow.jsonl")))
    blank_frames = [f for f, dets in enumerate(frames) if not dets]
    # At 10 fps: 2 s (20 frames) settle, 1 s (10 frames) gap, repeating.
    assert blank_frames == list(range(20, 30)) + list(range(50, 60))


def test_a_tracker_with_no_tracks_scores_zero_and_has_no_recovery_rate(tmp_path, dataset):
    script = tmp_path / "empty_tracker.py"
    script.write_text(
        "import sys\n"
        "lines = open(sys.argv[1], encoding='utf-8').read().splitlines()\n"
        "with open(sys.argv[2], 'w', encoding='utf-8', newline='\\n') as f:\n"
        "    for i, _ in enumerate(lines):\n"
        "        f.write('{\"frame\": %d, \"tracks\": []}\\n' % i)\n", encoding="utf-8")

    results = evaluate(dataset, tmp_path / "work", f'"{sys.executable}" "{script}"', "empty",
                       drop_rates=[0.2], gap_seconds=[1.0])

    assert results["random_deletion"][0]["id_kept"] == 0.0
    gap = results["gaps"][0]
    assert gap["gaps"] > 0 and gap["held_before"] == 0
    assert gap["recovery"] is None  # nothing to recover, so no rate, rather than a flattering one
    assert gap["prediction_error_px"] is None


def test_a_tracker_that_drops_frames_is_refused(tmp_path, dataset):
    script = tmp_path / "short_tracker.py"
    script.write_text(
        "import sys\n"
        "with open(sys.argv[2], 'w', encoding='utf-8', newline='\\n') as f:\n"
        "    f.write('{\"frame\": 0, \"tracks\": []}\\n')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1 frames"):
        evaluate(dataset, tmp_path / "work", f'"{sys.executable}" "{script}"', "short",
                 drop_rates=[0.2], gap_seconds=[])
