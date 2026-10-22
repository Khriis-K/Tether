"""Near-duplicate removal by perceptual hash (pHash), for the Drone-vs-Bird bird test images (ADR 0004).

pHash: the image in grayscale, shrunk to 32x32, its 2-D DCT, and the top-left
8x8 (lowest) frequencies, each set to 1 if above their median: 64 bits that
survive resizing, JPEG and small brightness changes. Two images are near
duplicates at a Hamming distance of MAX_DISTANCE bits or fewer.

dedup() walks the images in order and keeps one unless it's a near duplicate
of an image already kept, so the result doesn't depend on which drops chain.
"""

import re
from collections.abc import Sequence

import cv2
import numpy as np

# A common pHash cut for "the same picture". On the bird images, pairs this close were frames of one
# shot with the bird barely moved (docs/benchmarks/drone-transfer-sets.md).
MAX_DISTANCE = 10


def phash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return low > np.median(low)


def distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def dedup(hashes: Sequence[np.ndarray], max_distance: int = MAX_DISTANCE) -> tuple[list[int], dict[int, int]]:
    """The indices kept, and for each dropped index the kept one it matched (the first within range)."""
    kept, duplicate_of = [], {}
    for i, h in enumerate(hashes):
        match = next((k for k in kept if distance(h, hashes[k]) <= max_distance), None)
        if match is None:
            kept.append(i)
        else:
            duplicate_of[i] = match
    return kept, duplicate_of


def natural_key(name: str) -> list:
    """Sort key putting "BT (2)" before "BT (10)"."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]
