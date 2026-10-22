"""Dumping a detector's detections for a list of loose images (#45): the images are linked into
a numbered sequence the C++ pipeline reads like a video, so the same model and preprocessing run
on them as on Anti-UAV clips."""

import json
import sys

import cv2
import numpy as np
import pytest

from tether.formats import load_detections
from tether.image_detect import dump_images, main

# Stands in for tether-pipeline: called as `<command> --model M --det-min-score S --dump DETECTIONS
# PATTERN`, it writes one line per image of the numbered sequence, with the image's mean gray level
# as the score so the order can be checked, and logs each call.
STUB = """
import json, sys
from pathlib import Path
import cv2
args = sys.argv[1:]
dump, pattern = Path(args[args.index("--dump") + 1]), args[-1]
files = sorted(Path(pattern).parent.glob("*" + Path(pattern).suffix))
with open(Path(__file__).with_name("calls.log"), "a") as log:
    log.write(json.dumps(args) + "\\n")
with open(dump, "w") as f:
    for i in range({extra} + len(files)):
        level = float(cv2.imread(str(Path(pattern % min(i, len(files) - 1))), 0).mean()) / 255
        f.write(json.dumps({{"frame": i, "detections": [{{"box": [1, 2, 3, 4], "score": level, "class": 0}}]}}) + "\\n")
"""


@pytest.fixture
def images(tmp_path):
    """Three images of different gray levels and sizes, named out of numeric order."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name, level, (w, h) in [("100.jpg", 51, (64, 48)), ("7.jpg", 102, (32, 32)), ("BT (2).jpg", 204, (40, 20))]:
        cv2.imwrite(str(folder / name), np.full((h, w, 3), level, np.uint8))
    return folder


def stub(tmp_path, extra_frames=0):
    script = tmp_path / "stub.py"
    script.write_text(STUB.format(extra=extra_frames), encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


def calls(tmp_path):
    log = tmp_path / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_the_images_are_run_in_list_order_into_one_detection_file(tmp_path, images):
    names = ["BT (2).jpg", "100.jpg", "7.jpg"]

    out = dump_images(images, names, "birds", tmp_path / "work", "models/yolo26n_1280x736.onnx",
                      stub(tmp_path), 0.01)

    assert out == tmp_path / "work" / "detections" / "yolo26n_1280x736" / "birds"
    scores = [frame[0].score for frame in load_detections(out / "detections.jsonl")]
    assert scores == pytest.approx([204 / 255, 51 / 255, 102 / 255], abs=0.02)


def test_the_dump_names_model_floor_images_and_their_sizes(tmp_path, images):
    out = dump_images(images, ["7.jpg", "100.jpg"], "seraphim-test", tmp_path / "work", "m/yolo.onnx",
                      stub(tmp_path), 0.01)

    info = json.loads((out / "dump.json").read_text(encoding="utf-8"))
    assert info["model"] == "yolo"
    assert info["min_score"] == 0.01
    assert info["set"] == "seraphim-test"
    assert info["images"] == ["7.jpg", "100.jpg"]
    assert info["sizes"] == [[32, 32], [64, 48]]


def test_the_detector_is_given_the_model_score_floor_output_and_sequence(tmp_path, images):
    out = dump_images(images, ["7.jpg"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path), 0.01)

    (args,) = calls(tmp_path)
    assert args[args.index("--model") + 1] == "m/yolo.onnx"
    assert args[args.index("--det-min-score") + 1] == "0.01"
    assert args[args.index("--dump") + 1] == str(out / "detections.jsonl")
    assert args[-1].endswith("%06d.jpg")


def test_a_detection_file_with_the_wrong_number_of_images_is_refused(tmp_path, images):
    with pytest.raises(ValueError):
        dump_images(images, ["7.jpg", "100.jpg"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path, 1), 0.01)


def test_a_dump_already_complete_is_not_run_again(tmp_path, images):
    for _ in range(2):
        dump_images(images, ["7.jpg"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path), 0.01)

    assert len(calls(tmp_path)) == 1


def test_a_rerun_with_other_settings_into_the_same_folder_is_refused(tmp_path, images):
    dump_images(images, ["7.jpg"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path), 0.01)

    with pytest.raises(ValueError):
        dump_images(images, ["7.jpg", "100.jpg"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path), 0.01)


def test_a_list_mixing_file_types_is_refused(tmp_path, images):
    cv2.imwrite(str(images / "8.png"), np.zeros((8, 8, 3), np.uint8))

    with pytest.raises(ValueError):
        dump_images(images, ["7.jpg", "8.png"], "s", tmp_path / "work", "m/yolo.onnx", stub(tmp_path), 0.01)


def test_the_command_runs_the_kept_images_of_a_list_file(tmp_path, images):
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps({"kept": ["100.jpg", "7.jpg"], "dropped": ["BT (2).jpg"]}), encoding="utf-8")

    assert main([str(images), "--list", str(listing), "--set", "s", "--model", "m/yolo.onnx",
                 "--det-min-score", "0.01", "--detector", stub(tmp_path),
                 "--work-dir", str(tmp_path / "work")]) == 0

    info = json.loads((tmp_path / "work/detections/yolo/s/dump.json").read_text(encoding="utf-8"))
    assert info["images"] == ["100.jpg", "7.jpg"]
