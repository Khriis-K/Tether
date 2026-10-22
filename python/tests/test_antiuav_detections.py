"""tether-antiuav on a folder of real detections: the tracker is scored on a detector's output.

The detector's own misses are the dropout, so every frame with a labeled drone is
scored, and the same clips are also run on their labeled boxes, the ceiling the
detector's row is read against.
"""

import json
import sys

import pytest

from tether.antiuav import DRONE, evaluate, evaluate_detections, load_clip, main, to_detections
from tether.antiuav_split import save_split
from tether.formats import Detection, load_detections, save_detections
from test_antiuav import steady_drone, write_clip

TETHER = f'"{sys.executable}" -m tether.cli'


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "Anti-UAV300"
    write_clip(root / "test", "clip-a", steady_drone(120))
    flies_left = steady_drone(100, speed=-3.0)
    write_clip(root / "test", "clip-b", flies_left[:80] + [None] * 10 + flies_left[90:])
    return root


def labeled_clips(root):
    return [root / "test" / "clip-a", root / "test" / "clip-b"]


def write_dump(folder, frames_by_clip, model="some-model", split="val", min_score=0.01):
    """A detection folder as tether-antiuav-detect leaves it: one file per clip plus dump.json."""
    folder.mkdir(parents=True)
    for clip, frames in frames_by_clip.items():
        save_detections(folder / f"{clip}.jsonl", frames)
    (folder / "dump.json").write_text(json.dumps({"model": model, "min_score": min_score, "split": split}),
                                      encoding="utf-8")
    return folder


def copied_labels(root, tmp_path):
    clips = [load_clip(d) for d in labeled_clips(root)]
    return write_dump(tmp_path / "dets", {c.name: to_detections(c.boxes) for c in clips})


def test_a_folder_that_copies_the_labels_scores_the_same_as_the_labeled_boxes(tmp_path, dataset):
    labeled = evaluate(dataset, tmp_path / "labeled", TETHER, "tether", split="test", drop_rates=[0.0],
                       gap_seconds=[])

    results = evaluate_detections(labeled_clips(dataset), copied_labels(dataset, tmp_path), tmp_path / "work",
                                  TETHER, "tether")

    (baseline,) = labeled["random_deletion"]
    detector = results["detector"]
    assert (detector["kept"], detector["frames"], detector["id_kept"]) == (
        baseline["kept"], baseline["frames"], baseline["id_kept"])
    assert results["labeled_boxes"] == detector


def test_the_labeled_box_row_is_scored_on_the_same_clips_whatever_the_detector_did(tmp_path, dataset):
    clips = [load_clip(d) for d in labeled_clips(dataset)]
    nothing = write_dump(tmp_path / "dets", {c.name: [[] for _ in c.boxes] for c in clips})

    results = evaluate_detections(labeled_clips(dataset), nothing, tmp_path / "work", TETHER, "tether")

    # A detector that finds nothing keeps nothing: its misses are scored, not skipped.
    assert (results["detector"]["kept"], results["detector"]["frames"]) == (0, 210)
    assert results["detector"]["id_kept"] == 0.0
    assert (results["labeled_boxes"]["kept"], results["labeled_boxes"]["frames"]) == (206, 210)


def test_missed_frames_count_against_the_id(tmp_path, dataset):
    clips = [load_clip(d) for d in labeled_clips(dataset)]
    # Every other frame of clip-a missed: the track searches through them, but searching doesn't hold.
    frames = {c.name: to_detections([b if c.name != "clip-a" or f % 2 == 0 else None for f, b in enumerate(c.boxes)])
              for c in clips}

    results = evaluate_detections(labeled_clips(dataset), write_dump(tmp_path / "dets", frames), tmp_path / "work",
                                  TETHER, "tether")

    assert results["detector"]["frames"] == 210
    assert results["detector"]["kept"] < results["labeled_boxes"]["kept"]


def test_classes_are_ignored_so_a_drone_seen_as_two_classes_keeps_one_id(tmp_path, dataset):
    # The COCO model is scored class-agnostically: its "airplane" one frame and "bird" the next are one drone.
    clips = [load_clip(d) for d in labeled_clips(dataset)]
    frames = {c.name: [[Detection(box=b, score=0.9, cls=4 + f % 2)] if b else [] for f, b in enumerate(c.boxes)]
              for c in clips}

    results = evaluate_detections(labeled_clips(dataset), write_dump(tmp_path / "dets", frames), tmp_path / "work",
                                  TETHER, "tether")

    assert results["detector"]["kept"] == results["labeled_boxes"]["kept"] == 206


def test_the_tracker_sees_every_detection_at_drone_class(tmp_path, dataset):
    clips = [load_clip(d) for d in labeled_clips(dataset)]
    frames = {c.name: [[Detection(box=b, score=0.5, cls=7)] if b else [] for b in c.boxes] for c in clips}

    evaluate_detections(labeled_clips(dataset), write_dump(tmp_path / "dets", frames), tmp_path / "work", TETHER,
                        "tether")

    given = load_detections(tmp_path / "work" / "detections" / "some-model-val" / "clip-a.jsonl")
    assert {(d.cls, d.score) for dets in given for d in dets} == {(DRONE, 0.5)}


def test_results_name_the_model_and_its_score_floor_and_are_saved(tmp_path, dataset):
    dets = copied_labels(dataset, tmp_path)

    results = evaluate_detections(labeled_clips(dataset), dets, tmp_path / "work", TETHER, "tether-0.352")

    assert (results["model"], results["det_min_score"], results["split"]) == ("some-model", 0.01, "val")
    assert results["tracker"] == "tether-0.352" and results["tracker_command"] == TETHER
    assert results["clips"] == 2
    assert [(c["clip"], c["frames"]) for c in results["per_clip"]] == [("clip-a", 120), ("clip-b", 90)]
    saved = tmp_path / "work" / "trackers" / "tether-0.352" / "some-model-val" / "results.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == results


def test_a_detection_file_of_the_wrong_length_is_refused(tmp_path, dataset):
    clips = [load_clip(d) for d in labeled_clips(dataset)]
    frames = {c.name: to_detections(c.boxes) for c in clips}
    frames["clip-b"] = frames["clip-b"][:-1]

    with pytest.raises(ValueError, match="99 frames"):
        evaluate_detections(labeled_clips(dataset), write_dump(tmp_path / "dets", frames), tmp_path / "work",
                            TETHER, "tether")


def test_a_folder_without_dump_json_is_refused(tmp_path, dataset):
    dets = copied_labels(dataset, tmp_path)
    (dets / "dump.json").unlink()

    with pytest.raises(FileNotFoundError, match="dump.json"):
        evaluate_detections(labeled_clips(dataset), dets, tmp_path / "work", TETHER, "tether")
    assert not (tmp_path / "work").exists()


def test_the_tool_takes_the_clips_from_the_split_the_folder_was_dumped_for(tmp_path, dataset):
    write_clip(dataset / "train", "clip-c", steady_drone(50))
    split_file = tmp_path / "split.json"
    save_split(split_file, {"seed": 0, "clips": {"train": [], "val": ["test/clip-a", "train/clip-c"], "test": []},
                            "frames": {"test/clip-a": 120, "train/clip-c": 50}})
    clips = [load_clip(dataset / c) for c in ("test/clip-a", "train/clip-c")]
    dets = write_dump(tmp_path / "dets", {c.name: to_detections(c.boxes) for c in clips})
    work = tmp_path / "work"

    assert main([str(dataset), "--detections", str(dets), "--split-file", str(split_file),
                 "--work-dir", str(work), "--name", "t"]) == 0

    saved = json.loads((work / "trackers" / "t" / "some-model-val" / "results.json").read_text(encoding="utf-8"))
    assert [c["clip"] for c in saved["per_clip"]] == ["clip-a", "clip-c"]
    assert saved["detector"]["frames"] == 170


def test_the_tool_refuses_a_split_other_than_the_dumps(tmp_path, dataset):
    split_file = tmp_path / "split.json"
    save_split(split_file, {"seed": 0, "clips": {"train": [], "val": ["test/clip-a"], "test": ["test/clip-b"]},
                            "frames": {"test/clip-a": 120, "test/clip-b": 90}})
    clip = load_clip(dataset / "test" / "clip-a")
    dets = write_dump(tmp_path / "dets", {"clip-a": to_detections(clip.boxes)}, split="val")

    with pytest.raises(ValueError, match="dumped for val"):
        main([str(dataset), "--detections", str(dets), "--split-file", str(split_file), "--split", "test",
              "--work-dir", str(tmp_path / "work")])


def test_an_empty_split_is_refused_rather_than_scored_0_of_0(tmp_path, dataset):
    with pytest.raises(FileNotFoundError, match="no clips"):
        evaluate_detections([], copied_labels(dataset, tmp_path), tmp_path / "work", TETHER, "tether")
