"""Near-duplicate removal by perceptual hash (#45), for the Drone-vs-Bird bird test images."""

import cv2
import numpy as np

from tether.near_duplicates import MAX_DISTANCE, dedup, distance, natural_key, phash


def scene(seed, w=640, h=640):
    """A smooth random scene: noise blurred into blobs, so its low frequencies are its own."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 25)


def test_an_image_and_its_jpeg_brightened_and_resized_copies_are_near_duplicates():
    image = scene(0)
    copies = [cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_COLOR),
              cv2.convertScaleAbs(image, alpha=1.0, beta=12),
              cv2.resize(image, (320, 320))]

    for copy in copies:
        assert distance(phash(image), phash(copy)) <= MAX_DISTANCE


def test_different_scenes_are_not_near_duplicates():
    hashes = [phash(scene(s)) for s in range(10)]

    assert all(distance(a, b) > MAX_DISTANCE for i, a in enumerate(hashes) for b in hashes[i + 1:])


def test_the_hash_has_64_bits_and_an_image_is_at_distance_0_from_itself():
    h = phash(scene(3))

    assert h.shape == (64,)
    assert distance(h, h) == 0


def test_dedup_keeps_the_first_of_each_group_in_order_and_records_what_each_drop_matched():
    a, b = phash(scene(0)), phash(scene(1))
    a2 = phash(cv2.convertScaleAbs(scene(0), alpha=1.0, beta=8))

    kept, duplicate_of = dedup([a, b, a2, a], MAX_DISTANCE)

    assert kept == [0, 1]
    assert duplicate_of == {2: 0, 3: 0}


def test_dedup_compares_with_kept_images_only():
    # c is near b but not a; b is a drop of a; c is kept, since only kept images can absorb it.
    a = np.zeros(64, bool)
    b = a.copy()
    b[:8] = True  # 8 from a
    c = b.copy()
    c[8:16] = True  # 8 from b, 16 from a

    kept, duplicate_of = dedup([a, b, c], 10)

    assert kept == [0, 2]
    assert duplicate_of == {1: 0}


def test_names_sort_by_their_numbers():
    names = ["BT (10).jpg", "BT (2).jpg", "BT (1).jpg", "BT (100).jpg"]

    assert sorted(names, key=natural_key) == ["BT (1).jpg", "BT (2).jpg", "BT (10).jpg", "BT (100).jpg"]
