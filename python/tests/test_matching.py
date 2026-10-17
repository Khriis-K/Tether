import numpy as np
import pytest

from tether.matching import MAX_CENTER_DISTANCE, center_distance, iou, match, match_by_distance


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlapping_boxes():
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_of_zero_area_boxes_is_zero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_match_pairs_each_track_with_its_overlapping_detection():
    tracks = [(0, 0, 10, 10), (100, 100, 110, 110)]
    dets = [(101, 101, 111, 111), (1, 1, 11, 11)]

    assert match(tracks, dets) == [(0, 1), (1, 0)]


def test_match_finds_global_optimum_not_greedy():
    # Greedy takes the single best pair (track 0, det 0, IoU ~0.82), which
    # leaves track 1 with nothing it overlaps enough. The optimal assignment
    # gives up a little on track 0 to match both.
    tracks = [(0, 0, 10, 10), (4, 0, 14, 10)]
    dets = [(1, 0, 11, 10), (-3, 0, 7, 10)]

    assert match(tracks, dets) == [(0, 1), (1, 0)]


def test_match_keeps_every_valid_pair_when_a_one_pair_option_costs_less():
    # T0/D0 and T1/D1 both clear MIN_IOU (~0.33 each). The alternative pairs
    # T0/D1 (~0.82) with T1/D0, which don't overlap at all. Scoring that
    # invalid pair as an ordinary cost would make the alternative look cheaper
    # and lose a valid match.
    tracks = [(0, 0, 10, 10), (6, 0, 16, 10)]
    dets = [(-5, 0, 5, 10), (1, 0, 11, 10)]

    assert match(tracks, dets) == [(0, 0), (1, 1)]


def test_match_rejects_pairs_below_min_iou():
    # The only possible pair barely overlaps (IoU ~0.05), so it must not match.
    assert match([(0, 0, 10, 10)], [(9, 0, 19, 10)]) == []


def test_match_handles_empty_inputs():
    assert match([], [(0, 0, 1, 1)]) == []
    assert match([(0, 0, 1, 1)], []) == []


def test_center_distance_is_measured_in_track_box_sizes():
    # Centers (5, 5) and (35, 45): 50 px apart, track box side 10.
    assert center_distance((0, 0, 10, 10), (30, 40, 40, 50)) == pytest.approx(5.0)


def test_center_distance_uses_the_track_box_larger_side():
    # Track is 10 wide, 40 tall; centers 20 px apart.
    assert center_distance((0, 0, 10, 40), (20, 0, 30, 40)) == pytest.approx(0.5)


def test_match_by_distance_pairs_boxes_that_do_not_overlap():
    tracks = [(0, 0, 10, 10), (200, 0, 210, 10)]
    dets = [(230, 0, 240, 10), (25, 0, 35, 10)]

    assert match_by_distance(tracks, dets) == [(0, 1), (1, 0)]


def test_match_by_distance_rejects_pairs_beyond_max_distance():
    side = 10
    just_inside = MAX_CENTER_DISTANCE * side - 1
    just_outside = MAX_CENTER_DISTANCE * side + 1

    assert match_by_distance([(0, 0, 10, 10)], [(just_inside, 0, just_inside + 10, 10)]) == [(0, 0)]
    assert match_by_distance([(0, 0, 10, 10)], [(just_outside, 0, just_outside + 10, 10)]) == []


def test_match_by_distance_respects_allowed():
    allowed = np.array([[False]])

    assert match_by_distance([(0, 0, 10, 10)], [(5, 0, 15, 10)], allowed) == []
