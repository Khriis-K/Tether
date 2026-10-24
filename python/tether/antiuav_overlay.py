"""Spot Anti-UAV frames by their turret overlay, to drop them from Seraphim test (ADR 0004).

Seraphim merges 23 sources whose filenames don't survive, and Roboflow projects
often repackage Anti-UAV, so its test images are scanned for Anti-UAV's overlay.
Every Anti-UAV visible-light frame carries the same reticle: twelve 2-px line
segments at fixed places in the 1920x1080 frame (corner brackets and four
ticks around the center), each pixel drawn black or white to stand out from
what is behind it.

reticle_score() maps an image back to 1920x1080 (as is, and as a 16:9 frame
letterboxed into it, Seraphim's resize-and-pad) and measures, along each
segment, how much its line differs from the bands on both sides, in the same
direction. From that it takes the same measure at offsets across the segment,
so texture, edges and periodic patterns that light up everywhere cancel. The
score is the median over the twelve segments. The reticle is nearly symmetric
(within 3 px at 1920x1080), so a mirrored frame still scores high; a rotated,
sheared or cropped one doesn't.
"""

from collections.abc import Iterator

import cv2
import numpy as np

FRAME_W, FRAME_H = 1920, 1080
# (orientation, the line's center row or column, start, end): measured on Anti-UAV frames. The 2-px line
# covers rows (or columns) at - 1 and at; start and end run along it, end exclusive. The score samples
# BAND rows either side (at - 2 to at + 1), so a line shifted a pixel by resizing still falls inside.
RETICLE = (("h", 325, 578, 768), ("h", 325, 1152, 1346), ("h", 541, 384, 576), ("h", 541, 1344, 1536),
           ("h", 757, 576, 768), ("h", 757, 1152, 1344),
           ("v", 577, 326, 432), ("v", 577, 648, 758), ("v", 961, 216, 324), ("v", 961, 756, 864),
           ("v", 1345, 324, 432), ("v", 1345, 648, 756))
BAND = 2  # rows each side of the line's center sampled as "on the line", at 1920x1080
SIDE = (8, 12)  # the bands beside it: this many rows away
OFFSETS = tuple(d for step in range(14, 51, 4) for d in (step, -step))  # where the line isn't, for the background
# Real Anti-UAV frames, letterboxed into 640 or 416 squares and saved as JPEG, score 20 or more; no
# Seraphim test image scores above 0 (docs/benchmarks/drone-transfer-sets.md).
OVERLAY_THRESHOLD = 14.0
SERAPHIM_PAD = 114  # the gray Seraphim (and YOLO) pads with


def reticle_score(image: np.ndarray) -> float:
    """How strongly the reticle shows, in gray levels; about 0 for an image without it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return max(_view_score(view) for view in _views(gray))


def has_antiuav_overlay(image: np.ndarray) -> bool:
    return reticle_score(image) >= OVERLAY_THRESHOLD


def seraphim_like(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """A 16:9 frame resized to `width` and padded to a square, as Seraphim stores a wide image."""
    height = round(width * FRAME_H / FRAME_W)
    out = np.full((width, width, *frame.shape[2:]), SERAPHIM_PAD, np.uint8)
    top = (width - height) // 2
    out[top:top + height] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return out


def _views(gray: np.ndarray) -> Iterator[np.ndarray]:
    """The whole image, and the centered 16:9 strip a letterboxed frame would fill."""
    yield gray
    h, w = gray.shape
    strip = round(w * FRAME_H / FRAME_W)
    if strip < h:
        top = (h - strip) // 2
        yield gray[top:top + strip]


def _view_score(view: np.ndarray) -> float:
    g = cv2.resize(view, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    evidence = []
    for kind, at, start, end in RETICLE:
        lines = g if kind == "h" else g.T  # rows run along the segment
        start, end = start + (end - start) // 10, end - (end - start) // 10  # off the corners
        background = max(_line_contrast(lines, at + d, start, end) for d in OFFSETS)
        evidence.append(_line_contrast(lines, at, start, end) - background)
    return float(np.median(evidence))


def _line_contrast(lines: np.ndarray, at: int, start: int, end: int) -> float:
    """Mean over the segment's length of how far the line stands out from both sides, in one direction."""
    on = lines[at - BAND:at + BAND, start:end].mean(0)
    above = lines[at - SIDE[1]:at - SIDE[0], start:end].mean(0)
    below = lines[at + SIDE[0]:at + SIDE[1], start:end].mean(0)
    up, down = on - above, on - below
    return float(np.where(up * down > 0, np.minimum(np.abs(up), np.abs(down)), 0.0).mean())
