import numpy as np
import pytest

from tether.detector import decode, letterbox
from tether.formats import Detection


def test_letterbox_scales_the_long_side_to_fit_and_pads_the_short_side_evenly():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    image, scale, (pad_x, pad_y) = letterbox(frame, size=640)

    assert image.shape == (640, 640, 3)
    assert scale == 1.0
    assert (pad_x, pad_y) == (0, 80)


def test_letterbox_puts_the_frame_between_gray_bars():
    frame = np.full((480, 640, 3), 7, dtype=np.uint8)

    image, _, _ = letterbox(frame, size=640)

    assert (image[:80] == 114).all()
    assert (image[80:560] == 7).all()
    assert (image[560:] == 114).all()


def test_letterbox_shrinks_frames_bigger_than_the_model_input():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    image, scale, (pad_x, pad_y) = letterbox(frame, size=640)

    assert image.shape == (640, 640, 3)
    assert scale == 0.5
    assert (pad_x, pad_y) == (0, 140)


def test_decode_maps_boxes_back_to_frame_pixels():
    # Frame 1280x720 letterboxed to 640: scale 0.5, 140 px of padding on top.
    output = np.array([[[100.0, 190.0, 200.0, 290.0, 0.9, 0.0]]], dtype=np.float32)

    dets = decode(output, scale=0.5, pad=(0, 140), frame_shape=(720, 1280), min_score=0.1)

    assert dets == [Detection(box=(200.0, 100.0, 400.0, 300.0), score=pytest.approx(0.9), cls=0)]


def test_decode_keeps_scores_at_the_threshold_and_drops_ones_below():
    output = np.array([[
        [10.0, 10.0, 20.0, 20.0, 0.1, 0.0],
        [10.0, 10.0, 20.0, 20.0, 0.09, 0.0],
        [30.0, 30.0, 40.0, 40.0, 0.5, 2.0],
    ]], dtype=np.float32)

    dets = decode(output, scale=1.0, pad=(0, 0), frame_shape=(100, 100), min_score=0.1)

    assert [(d.score, d.cls) for d in dets] == [(pytest.approx(0.1), 0), (pytest.approx(0.5), 2)]


def test_decode_clips_boxes_to_the_frame_and_drops_ones_left_empty():
    output = np.array([[
        [-5.0, -5.0, 50.0, 120.0, 0.9, 0.0],   # pokes out top-left and bottom
        [150.0, 10.0, 160.0, 20.0, 0.9, 0.0],  # entirely right of the frame
    ]], dtype=np.float32)

    dets = decode(output, scale=1.0, pad=(0, 0), frame_shape=(100, 100), min_score=0.1)

    assert [d.box for d in dets] == [(0.0, 0.0, 50.0, 100.0)]


def test_decode_returns_plain_python_numbers():
    output = np.array([[[10.0, 10.0, 20.0, 20.0, 0.9, 3.0]]], dtype=np.float32)

    det = decode(output, scale=1.0, pad=(0, 0), frame_shape=(100, 100), min_score=0.1)[0]

    assert all(type(v) is float for v in det.box)
    assert type(det.score) is float
    assert type(det.cls) is int
