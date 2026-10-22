"""Building the two outside test sets (#45): Seraphim test with Anti-UAV frames dropped, and the
Drone-vs-Bird bird test images with near-duplicates removed, ground truth from the filename."""

import json

import cv2
import numpy as np
import pytest

from tether.antiuav_overlay import OVERLAY_THRESHOLD, RETICLE
from tether.transfer_sets import TRANSFORMS, dedup_birds, main, overlay_check, scan_seraphim


def frame(seed=0, reticle=False):
    rng = np.random.default_rng(seed)
    gray = np.clip(np.linspace(90, 150, 1920)[None, :] + rng.normal(0, 8, (1080, 1920)), 0, 255).astype(np.uint8)
    if reticle:
        for kind, at, start, end in RETICLE:
            if kind == "h":
                gray[at - 1:at + 1, start:end] = 255
            else:
                gray[start:end, at - 1:at + 1] = 255
    return gray


def scene(seed):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (320, 320, 3), dtype=np.uint8), (0, 0), 15)


@pytest.fixture
def seraphim(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for name, image in [("10.jpg", frame(1)), ("9.jpg", TRANSFORMS["letterboxed 640, JPEG 75"](frame(2, True))),
                        ("100.jpg", scene(3))]:
        cv2.imwrite(str(folder / name), image)
    return folder


def test_seraphim_images_with_the_overlay_are_dropped_and_listed_with_their_score(seraphim):
    result = scan_seraphim(seraphim)

    assert result["images"] == 3
    assert [d["image"] for d in result["dropped"]] == ["9.jpg"]
    assert result["dropped"][0]["score"] >= OVERLAY_THRESHOLD
    assert result["kept"] == ["10.jpg", "100.jpg"]
    assert (result["kept_count"], result["dropped_count"]) == (2, 1)
    assert result["threshold"] == OVERLAY_THRESHOLD


def test_the_highest_scoring_kept_images_are_listed_for_checking(seraphim):
    highest = scan_seraphim(seraphim)["highest_kept"]

    assert {h["image"] for h in highest} == {"10.jpg", "100.jpg"}
    assert highest[0]["score"] >= highest[1]["score"]


@pytest.fixture
def birds(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    near_copy = cv2.convertScaleAbs(scene(1), alpha=1.0, beta=10)
    # By number BT (2) comes before BT (10) and is kept; sorted as strings, BT (10) would be.
    for name, image in [("BT (1).jpg", scene(0)), ("BT (2).jpg", near_copy), ("BT (10).jpg", scene(1)),
                        ("DT (1).jpg", scene(2)), ("BT (3).jpg", scene(0))]:
        cv2.imwrite(str(folder / name), image)
    return folder


def test_only_bird_images_are_taken_and_near_duplicates_dropped_in_name_order(birds):
    result = dedup_birds(birds)

    assert result["images"] == 4  # the BT images; DT (drone) isn't one
    assert result["kept"] == ["BT (1).jpg", "BT (2).jpg"]
    assert result["duplicates"] == {"BT (3).jpg": "BT (1).jpg", "BT (10).jpg": "BT (2).jpg"}
    assert result["kept_count"] == 2


def test_ground_truth_is_the_filename_so_label_files_are_never_read(birds, tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "BT (1).txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # mislabeled "Drone"

    assert dedup_birds(birds)["kept"] == ["BT (1).jpg", "BT (2).jpg"]


def test_the_overlay_check_scores_every_frame_under_every_transform():
    result = overlay_check([frame(1, True), frame(2, True)])

    assert result["frames"] == 2
    assert set(result["transforms"]) == set(TRANSFORMS)
    for row in result["transforms"].values():
        assert row["flagged"] == 2
        assert row["min_score"] >= OVERLAY_THRESHOLD


def test_the_overlay_check_counts_frames_it_misses():
    result = overlay_check([frame(1, True), frame(2)])

    assert all(row["flagged"] == 1 for row in result["transforms"].values())


def test_the_seraphim_command_writes_its_list(seraphim, tmp_path):
    out = tmp_path / "seraphim.json"

    assert main(["seraphim", str(seraphim), "--out", str(out)]) == 0

    assert json.loads(out.read_text(encoding="utf-8"))["kept"] == ["10.jpg", "100.jpg"]


def test_the_birds_command_writes_its_list(birds, tmp_path):
    out = tmp_path / "birds.json"

    assert main(["birds", str(birds), "--out", str(out)]) == 0

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["kept"] == ["BT (1).jpg", "BT (2).jpg"]
    assert saved["max_distance"] == 10
