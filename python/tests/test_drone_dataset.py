"""The drone training set (#43, ADR 0004): every Nth frame of the split's train clips as a YOLO dataset."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tether.antiuav_split import session_of
from tether.drone_dataset import build_dataset, main, sampled_frames, yolo_label

WIDTH, HEIGHT = 64, 36  # the fake clips' frame size


def write_clip(root: Path, clip: str, exist: list[int]) -> None:
    """A tiny Anti-UAV clip: `visible.json` with a 10x8 box where the drone exists, and a matching video."""
    d = root / clip
    d.mkdir(parents=True)
    rects = [[4 + i % 20, 6, 10, 8] if e else [] for i, e in enumerate(exist)]
    (d / "visible.json").write_text(json.dumps({"exist": exist, "gt_rect": rects}), encoding="utf-8")
    video = cv2.VideoWriter(str(d / "visible.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH, HEIGHT))
    for i in range(len(exist)):
        video.write(np.full((HEIGHT, WIDTH, 3), i % 256, np.uint8))
    video.release()


def fake_dataset(root: Path, frames: int = 95, absent_rate: float = 0.0, seed: int = 0) -> dict:
    """Two train sessions, one val session (in the official train folder, as in the real split) and one
    test session, with the split file that names them."""
    rng = np.random.default_rng(seed)
    clips = {"train": ["train/20190925_100000_1_1", "val/20190925_100000_1_2", "train/20190926_110000_1_1"],
             "val": ["train/20190925_120000_1_1"],
             "test": ["test/20190926_130000_1_1"]}
    for clip in sum(clips.values(), []):
        write_clip(root, clip, [int(r >= absent_rate) for r in rng.random(frames)])
    return {"seed": 0, "clips": clips, "frames": {c: frames for c in sum(clips.values(), [])}}


def image_names(out: Path, split: str) -> list[str]:
    return sorted(p.stem for p in (out / "images" / split).iterdir())


def test_every_stride_th_frame_is_sampled_from_the_first():
    assert list(sampled_frames(25, stride=10)) == [0, 10, 20]
    assert list(sampled_frames(3, stride=1)) == [0, 1, 2]


@pytest.mark.parametrize("stride", [0, -1])
def test_a_stride_below_one_is_refused(stride):
    with pytest.raises(ValueError):
        sampled_frames(25, stride=stride)


def test_a_yolo_label_is_the_box_center_and_size_over_the_frame_size():
    assert yolo_label((10.0, 20.0, 30.0, 60.0), 100, 200) == "0 0.200000 0.200000 0.200000 0.200000"


def test_an_absent_drone_has_an_empty_label():
    assert yolo_label(None, 100, 200) == ""


def test_a_box_over_the_frame_edge_is_clipped_to_the_frame():
    assert yolo_label((-10.0, 150.0, 30.0, 250.0), 100, 200) == "0 0.150000 0.875000 0.300000 0.250000"


def test_a_box_wholly_outside_the_frame_has_an_empty_label():
    assert yolo_label((110.0, 20.0, 130.0, 60.0), 100, 200) == ""


def test_the_training_images_come_only_from_the_splits_train_sessions(tmp_path):
    split = fake_dataset(tmp_path / "antiuav")

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    train_sessions = {session_of(Path(c).name) for c in split["clips"]["train"]}
    held_out = {session_of(Path(c).name) for c in split["clips"]["val"] + split["clips"]["test"]}
    used = {session_of(name.rsplit("_", 1)[0]) for name in image_names(tmp_path / "out", "train")}
    assert used == train_sessions
    assert not used & held_out


def test_val_images_come_only_from_the_splits_val_sessions_and_test_is_never_read(tmp_path):
    split = fake_dataset(tmp_path / "antiuav")
    for f in (tmp_path / "antiuav" / "test").rglob("*"):
        if f.is_file():
            f.unlink()  # a read of test would now fail

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    used = {session_of(name.rsplit("_", 1)[0]) for name in image_names(tmp_path / "out", "val")}
    assert used == {"20190925_120000"}
    assert not (tmp_path / "out" / "images" / "test").exists()


def test_each_clip_gives_its_stride_th_frames_with_a_label_file_each(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=95)

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    names = image_names(tmp_path / "out", "train")
    assert len(names) == 3 * 10  # frames 0, 10, ..., 90 of each of the 3 train clips
    assert "20190925_100000_1_2_000090" in names
    labels = sorted(p.stem for p in (tmp_path / "out" / "labels" / "train").iterdir())
    assert labels == names


def test_an_images_label_is_its_frames_box(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=25)

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    label = (tmp_path / "out" / "labels" / "train" / "20190925_100000_1_1_000010.txt").read_text()
    # frame 10's box: x 14, y 6, 10x8 px in a 64x36 frame
    assert label.strip() == yolo_label((14.0, 6.0, 24.0, 14.0), WIDTH, HEIGHT)


def test_images_are_resized_to_the_given_width_keeping_the_aspect_ratio(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=5)

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=32)

    image = cv2.imread(str(tmp_path / "out" / "images" / "train" / "20190925_100000_1_1_000000.jpg"))
    assert image.shape[:2] == (18, 32)


def test_the_drone_absent_share_matches_the_source(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=2000, absent_rate=0.05)

    summary = build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    train = summary["train"]
    exist = sum((json.loads((tmp_path / "antiuav" / c / "visible.json").read_text(encoding="utf-8"))["exist"]
                 for c in split["clips"]["train"]), [])
    assert train["source_absent_share"] == pytest.approx(exist.count(0) / len(exist))
    assert train["images"] == 3 * 200
    assert train["absent_images"] == sum(
        1 for p in (tmp_path / "out" / "labels" / "train").iterdir() if not p.read_text().strip())
    assert abs(train["absent_images"] / train["images"] - train["source_absent_share"]) < 0.02
    assert 0.03 < train["source_absent_share"] < 0.07


def test_the_dataset_file_names_one_class_drone_and_the_two_image_folders(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=5)

    build_dataset(tmp_path / "antiuav", split, tmp_path / "out", stride=10, width=WIDTH)

    data = (tmp_path / "out" / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data and "val: images/val" in data
    assert "names:\n  0: drone" in data


def test_the_cli_writes_the_settings_beside_the_dataset(tmp_path):
    split = fake_dataset(tmp_path / "antiuav", frames=25)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split), encoding="utf-8")

    assert main([str(tmp_path / "antiuav"), str(tmp_path / "out"), "--split-file", str(split_file),
                 "--stride", "5", "--width", str(WIDTH)]) == 0

    info = json.loads((tmp_path / "out" / "dataset.json").read_text(encoding="utf-8"))
    assert info["stride"] == 5 and info["width"] == WIDTH and info["split_seed"] == 0
    assert info["train"]["images"] == 3 * 5
