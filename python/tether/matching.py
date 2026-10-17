"""Pairing detections with tracks: IoU cost, Hungarian assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tether.formats import Box

MIN_IOU = 0.3  # pairs that overlap less than this are never matched

# Cost of a pair that can't match. It exceeds any total of valid costs (each is
# at most 1), so the solver first maximizes the number of valid pairs.
_INVALID_COST = 1e6


def iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def match(track_boxes: list[Box], det_boxes: list[Box],
          allowed: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Best one-to-one pairing of tracks to detections, as (track, detection) index pairs.

    A pair is valid when its IoU is at least MIN_IOU and `allowed` (an optional
    boolean grid) permits it. Returns the assignment with the most valid pairs,
    and among those the lowest total (1 - IoU). Result is sorted by track index.
    """
    if not track_boxes or not det_boxes:
        return []
    ious = np.array([[iou(t, d) for d in det_boxes] for t in track_boxes])
    valid = ious >= MIN_IOU
    if allowed is not None:
        valid &= allowed
    rows, cols = linear_sum_assignment(np.where(valid, 1.0 - ious, _INVALID_COST))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
