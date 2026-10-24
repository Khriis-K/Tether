"""The low-score band moves with the cutoff: its floor is low_score_ratio times min_score.

At the default cutoff the floor is ByteTrack's 0.1, so nothing changes there. A
model run at another threshold gets a band in proportion, rather than one that is
empty (cutoff at or under 0.1) or several times wider (a high cutoff).
"""

import pytest

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections
from tether.tracker import LOW_SCORE_RATIO, MIN_SCORE, Tracker

BOX = (0.0, 50.0, 20.0, 70.0)  # a still object, so every prediction is this box
HIGH = 0.9


def det(score):
    return Detection(box=BOX, score=score, cls=0)


def status_after(tracker, score):
    """Confirm a track, then give it one detection at `score`; the track's status after it."""
    for _ in range(3):
        tracker.update([det(HIGH)])
    (track,) = tracker.update([det(score)])
    return track.status


def test_at_the_default_cutoff_the_floor_is_bytetracks_0_1_exactly():
    assert LOW_SCORE_RATIO * MIN_SCORE == 0.1


@pytest.mark.parametrize("score, status", [(0.1, "searching"), (0.1001, "tracked"), (0.2499, "tracked")])
def test_at_the_default_cutoff_the_band_is_0_1_to_0_25(score, status):
    assert status_after(Tracker(confirm_frames=3), score) == status


@pytest.mark.parametrize("score, status", [
    (0.15, "searching"),  # in the old fixed band, but under the moved floor (0.2)
    (0.2, "searching"),  # the floor itself is out, as at the default
    (0.21, "tracked"),
    (0.49, "tracked"),
])
def test_a_higher_cutoff_moves_the_floor_up_in_proportion(score, status):
    assert status_after(Tracker(confirm_frames=3, min_score=0.5), score) == status


@pytest.mark.parametrize("score, status", [(0.03, "searching"), (0.05, "tracked")])
def test_a_cutoff_under_0_1_still_has_a_band(score, status):
    # A fixed 0.1 floor would leave no band at all under a 0.1 cutoff.
    assert status_after(Tracker(confirm_frames=3, min_score=0.1), score) == status


def test_the_ratio_can_be_set():
    assert status_after(Tracker(confirm_frames=3, min_score=0.5, low_score_ratio=0.2), 0.15) == "tracked"
    assert status_after(Tracker(confirm_frames=3, min_score=0.5, low_score_ratio=0.2), 0.1) == "searching"


def test_a_ratio_of_one_turns_the_low_pass_off():
    assert status_after(Tracker(confirm_frames=3, min_score=0.5, low_score_ratio=1.0), 0.49) == "searching"


def run_cli(tmp_path, score, *options):
    dets, tracks = tmp_path / "dets.jsonl", tmp_path / "tracks.jsonl"
    save_detections(dets, [[det(HIGH)]] * 3 + [[det(score)]])
    assert main([str(dets), str(tracks), *options]) == 0
    return load_tracks(tracks)[-1][0].status


def test_the_cli_moves_the_band_with_min_score(tmp_path):
    assert run_cli(tmp_path, 0.15, "--min-score", "0.5") == "searching"
    assert run_cli(tmp_path, 0.25, "--min-score", "0.5") == "tracked"


def test_the_cli_sets_the_ratio(tmp_path):
    assert run_cli(tmp_path, 0.15, "--min-score", "0.5", "--low-score-ratio", "0.2") == "tracked"
