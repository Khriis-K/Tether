"""The session-disjoint Anti-UAV split (#42, ADR 0004): whole recording sessions held out as val."""

import json
from pathlib import Path

import pytest

from tether.antiuav_split import (MAX_VAL_FRACTION, MIN_VAL_FRACTION, clip_dirs, count_frames, load_split, main,
                                  make_split, save_split, session_of)

COMMITTED_SPLIT = Path(__file__).resolve().parents[2] / "splits" / "antiuav.json"


def official(day1_sessions=12, day2_sessions=10, clips_per_session=3, frames=1000):
    """Frame counts per clip, as {official split: {clip name: frames}}, the way Anti-UAV300 deals
    one session's clips out to both train and val. Test has sessions of its own. Each session
    is about 4.5% of the pooled frames, so four of them land in the 15-20% band."""
    counts = {"train": {}, "val": {}, "test": {}}
    sessions = ([f"20190925_1{i:02d}000" for i in range(day1_sessions)]
                + [f"20190926_1{i:02d}000" for i in range(day2_sessions)])
    for s in sessions:
        for k in range(1, clips_per_session + 1):
            counts["val" if k == 2 else "train"][f"{s}_1_{k}"] = frames + k
    counts["test"] = {"20190925_230000_1_1": 900, "20190926_230000_1_1": 800}
    return counts


def split_sessions(split, name):
    return {session_of(Path(c).name) for c in split["clips"][name]}


def pooled_frames(split, name):
    return sum(split["frames"][c] for c in split["clips"][name])


def test_a_session_is_the_clip_name_without_its_last_two_fields():
    assert session_of("20190925_101846_1_4") == "20190925_101846"


def test_no_session_is_in_two_splits():
    split = make_split(official(), seed=0)

    train, val, test = (split_sessions(split, n) for n in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test


def test_test_is_exactly_the_official_test():
    split = make_split(official(), seed=0)

    assert split["clips"]["test"] == ["test/20190925_230000_1_1", "test/20190926_230000_1_1"]


def test_train_and_val_together_are_the_pooled_official_train_and_val():
    counts = official()

    split = make_split(counts, seed=0)

    pooled = {f"{s}/{c}" for s in ("train", "val") for c in counts[s]}
    assert set(split["clips"]["train"]) | set(split["clips"]["val"]) == pooled
    assert not set(split["clips"]["train"]) & set(split["clips"]["val"])


def test_val_holds_15_to_20_percent_of_pooled_frames():
    for seed in range(20):
        split = make_split(official(), seed=seed)
        train, val = pooled_frames(split, "train"), pooled_frames(split, "val")
        assert MIN_VAL_FRACTION <= val / (train + val) <= MAX_VAL_FRACTION


def test_val_has_sessions_from_both_recording_days():
    for seed in range(20):
        days = {s[:8] for s in split_sessions(make_split(official(), seed=seed), "val")}
        assert days == {"20190925", "20190926"}


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one():
    assert make_split(official(), seed=3) == make_split(official(), seed=3)
    assert len({tuple(make_split(official(), seed=s)["clips"]["val"]) for s in range(10)}) > 1


def test_the_split_records_its_seed_and_every_clips_frame_count():
    counts = official()

    split = make_split(counts, seed=7)

    assert split["seed"] == 7
    assert split["frames"]["train/20190925_100000_1_1"] == counts["train"]["20190925_100000_1_1"]
    assert split["frames"]["test/20190926_230000_1_1"] == 800


def test_a_split_that_cannot_land_in_range_is_refused():
    # Two sessions: holding out either one is half the frames.
    counts = {"train": {"20190925_100000_1_1": 1000, "20190926_100000_1_1": 1000}, "val": {}, "test": {}}

    with pytest.raises(ValueError):
        make_split(counts, seed=0)


def test_saved_split_loads_back_and_names_clip_folders(tmp_path):
    split = make_split(official(), seed=0)
    path = tmp_path / "split.json"

    save_split(path, split)

    assert load_split(path) == split
    assert clip_dirs("data/Anti-UAV300", split, "test") == [Path("data/Anti-UAV300/test/20190925_230000_1_1"),
                                                            Path("data/Anti-UAV300/test/20190926_230000_1_1")]


def test_frames_are_counted_from_the_labels(tmp_path):
    for split, clip, n in [("train", "20190925_100000_1_1", 3), ("val", "20190925_100000_1_2", 2),
                           ("test", "20190926_100000_1_1", 4)]:
        (tmp_path / split / clip).mkdir(parents=True)
        (tmp_path / split / clip / "visible.json").write_text(
            json.dumps({"exist": [1] * n, "gt_rect": [[1, 2, 3, 4]] * n}), encoding="utf-8")

    assert count_frames(tmp_path) == {"train": {"20190925_100000_1_1": 3}, "val": {"20190925_100000_1_2": 2},
                                      "test": {"20190926_100000_1_1": 4}}


def test_the_command_writes_the_split_file(tmp_path, monkeypatch):
    counts = official()
    monkeypatch.setattr("tether.antiuav_split.count_frames", lambda root: counts)
    out = tmp_path / "split.json"

    assert main(["data/Anti-UAV300", "--seed", "5", "--out", str(out)]) == 0

    assert load_split(out) == make_split(counts, seed=5)


# --- The committed split file every drone ticket uses ---

def test_the_committed_split_is_reproduced_from_its_own_seed_and_frame_counts():
    split = load_split(COMMITTED_SPLIT)
    counts = {"train": {}, "val": {}, "test": {}}
    for clip, n in split["frames"].items():
        official_split, name = clip.split("/")
        counts[official_split][name] = n

    assert make_split(counts, seed=split["seed"]) == split


def test_the_committed_split_is_session_disjoint_with_val_from_both_days():
    split = load_split(COMMITTED_SPLIT)

    train, val, test = (split_sessions(split, n) for n in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert {s[:8] for s in val} == {"20190925", "20190926"}
    assert MIN_VAL_FRACTION <= pooled_frames(split, "val") / (
        pooled_frames(split, "train") + pooled_frames(split, "val")) <= MAX_VAL_FRACTION
    assert all(c.startswith("test/") for c in split["clips"]["test"])
    assert len(split["clips"]["test"]) == 91 and len(split["clips"]["train"]) + len(split["clips"]["val"]) == 227
