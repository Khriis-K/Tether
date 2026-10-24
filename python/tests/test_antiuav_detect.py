"""Dumping a detector's detections for every clip of an Anti-UAV split (#42), in the detection file
format the tracker reads."""

import json
import sys

import pytest

from tether.antiuav_detect import dump_split, main
from tether.formats import load_detections

# Stands in for tether-demo / tether-pipeline: called as `<command> --model M --det-min-score S
# --dump DETECTIONS VIDEO`, it writes one detection per frame (as many frames as the clip's labels)
# and logs each call.
STUB = """
import json, sys
from pathlib import Path
args = sys.argv[1:]
dump, video = Path(args[args.index("--dump") + 1]), Path(args[-1])
n = len(json.loads((video.parent / "visible.json").read_text())["exist"]) + {extra}
with open(Path(__file__).with_name("calls.log"), "a") as log:
    log.write(json.dumps(args) + "\\n")
with open(dump, "w") as f:
    for i in range(n):
        f.write(json.dumps({{"frame": i, "detections": [{{"box": [1, 2, 3, 4], "score": 0.5, "class": 4}}]}}) + "\\n")
"""


@pytest.fixture
def dataset(tmp_path):
    """Two val clips and one train clip, each a visible.json beside a (dummy) visible.mp4."""
    root = tmp_path / "Anti-UAV300"
    frames = {"val/20190925_100000_1_2": 3, "train/20190926_100000_1_1": 2, "val/20190926_100000_1_2": 4}
    for clip, n in frames.items():
        (root / clip).mkdir(parents=True)
        (root / clip / "visible.json").write_text(json.dumps({"exist": [1] * n}), encoding="utf-8")
        (root / clip / "visible.mp4").write_bytes(b"")
    split = {"seed": 0, "frames": frames,
             "clips": {"train": ["val/20190925_100000_1_2"],
                       "val": ["train/20190926_100000_1_1", "val/20190926_100000_1_2"], "test": []}}
    return root, split


def stub(tmp_path, extra_frames=0):
    script = tmp_path / "stub.py"
    script.write_text(STUB.format(extra=extra_frames), encoding="utf-8")
    return f'"{sys.executable}" "{script}"'


def calls(tmp_path):
    log = tmp_path / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_every_clip_of_the_split_gets_a_detection_file(tmp_path, dataset):
    root, split = dataset

    out = dump_split(root, split, "val", tmp_path / "work", "models/yolo26n_1280x736.onnx", stub(tmp_path), 0.1)

    assert out == tmp_path / "work" / "detections" / "yolo26n_1280x736" / "val"
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["20190926_100000_1_1.jsonl", "20190926_100000_1_2.jsonl"]
    assert len(load_detections(out / "20190926_100000_1_2.jsonl")) == 4


def test_the_detector_is_given_the_model_score_floor_output_and_clip_video(tmp_path, dataset):
    root, split = dataset

    out = dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.05)

    assert calls(tmp_path)[0] == ["--model", "models/m.onnx", "--det-min-score", "0.05",
                                  "--dump", str(out / "20190926_100000_1_1.jsonl"),
                                  str(root / "train" / "20190926_100000_1_1" / "visible.mp4")]


def test_the_dump_records_the_model_and_score_floor(tmp_path, dataset):
    root, split = dataset

    out = dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)

    info = json.loads((out / "dump.json").read_text(encoding="utf-8"))
    assert (info["model"], info["model_path"], info["min_score"], info["split"]) == ("m", "models/m.onnx", 0.1, "val")


def test_clips_already_dumped_in_full_are_skipped(tmp_path, dataset):
    root, split = dataset
    dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)
    out = tmp_path / "work" / "detections" / "m" / "val"
    (out / "20190926_100000_1_2.jsonl").write_text('{"frame": 0, "detections": []}\n')  # cut short

    dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)

    assert [c[-1] for c in calls(tmp_path)].count(str(root / "train" / "20190926_100000_1_1" / "visible.mp4")) == 1
    assert len(load_detections(out / "20190926_100000_1_2.jsonl")) == 4


def test_a_detection_file_with_the_wrong_number_of_frames_is_refused(tmp_path, dataset):
    root, split = dataset

    with pytest.raises(ValueError):
        dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path, extra_frames=1), 0.1)


def test_the_command_dumps_the_named_split(tmp_path, dataset):
    root, split = dataset
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split), encoding="utf-8")

    assert main([str(root), "--model", "models/m.onnx", "--split-file", str(split_file), "--split", "train",
                 "--detector", stub(tmp_path), "--work-dir", str(tmp_path / "work")]) == 0

    assert [p.name for p in (tmp_path / "work" / "detections" / "m" / "train").glob("*.jsonl")] == [
        "20190925_100000_1_2.jsonl"]


def test_a_clip_cut_short_mid_line_by_a_stopped_run_is_dumped_again(tmp_path, dataset):
    root, split = dataset
    dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)
    out = tmp_path / "work" / "detections" / "m" / "val"
    (out / "20190926_100000_1_2.jsonl").write_text('{"frame": 0, "detections": []}\n{"frame": 1, "dete')

    dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)

    assert len(load_detections(out / "20190926_100000_1_2.jsonl")) == 4


def test_a_rerun_with_other_settings_into_the_same_folder_is_refused(tmp_path, dataset):
    # The old files would be kept and labeled with the new score floor.
    root, split = dataset
    dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.1)

    with pytest.raises(ValueError):
        dump_split(root, split, "val", tmp_path / "work", "models/m.onnx", stub(tmp_path), 0.05)
