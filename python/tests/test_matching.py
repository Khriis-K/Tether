import pytest

from tether.matching import iou, match


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


def test_match_rejects_pairs_below_min_iou():
    # The only possible pair barely overlaps (IoU ~0.05), so it must not match.
    assert match([(0, 0, 10, 10)], [(9, 0, 19, 10)]) == []


def test_match_handles_empty_inputs():
    assert match([], [(0, 0, 1, 1)]) == []
    assert match([(0, 0, 1, 1)], []) == []
