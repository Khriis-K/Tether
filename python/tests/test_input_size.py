"""The detector's input size comes from the model, and may be a rectangle (#21)."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tether.detector import OnnxDetector, decode, letterbox, model_input_size, prepare
from tether.formats import Detection


def test_letterbox_fits_a_wide_frame_into_a_rectangular_input():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    image, scale, (pad_x, pad_y) = letterbox(frame, size=(736, 1280))

    assert image.shape == (736, 1280, 3)
    assert scale == pytest.approx(2 / 3)
    assert (pad_x, pad_y) == (0, 8)


def test_letterbox_pads_a_rectangular_input_top_and_bottom_with_gray():
    frame = np.full((1080, 1920, 3), 7, dtype=np.uint8)

    image, _, _ = letterbox(frame, size=(736, 1280))

    assert (image[:8] == 114).all()
    assert (image[8:728] == 7).all()
    assert (image[728:] == 114).all()


def test_letterbox_pads_the_sides_when_the_frame_is_narrower_than_the_input():
    frame = np.zeros((736, 640, 3), dtype=np.uint8)

    image, scale, (pad_x, pad_y) = letterbox(frame, size=(736, 1280))

    assert image.shape == (736, 1280, 3)
    assert scale == 1.0
    assert (pad_x, pad_y) == (320, 0)


def test_decode_maps_boxes_from_a_rectangular_input_back_to_frame_pixels():
    # 1920x1080 frame letterboxed to 1280x736: scale 2/3, 8 px of padding on top.
    output = np.array([[[100.0, 108.0, 300.0, 308.0, 0.9, 0.0]]], dtype=np.float32)

    dets = decode(output, scale=2 / 3, pad=(0, 8), frame_shape=(1080, 1920), min_score=0.1)

    assert dets == [Detection(box=pytest.approx((150.0, 150.0, 450.0, 450.0)), score=pytest.approx(0.9), cls=0)]


def test_prepare_builds_a_blob_of_the_requested_height_and_width():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    prepared = prepare(frame, (736, 1280))

    assert prepared.blob.shape == (1, 3, 736, 1280)
    assert prepared.blob.dtype == np.float32
    assert prepared.pad == (0, 8)
    assert prepared.frame_shape == (1080, 1920)


def test_model_input_size_reads_height_and_width_from_an_nchw_shape():
    assert model_input_size([1, 3, 736, 1280]) == (736, 1280)


@pytest.mark.parametrize("shape", [[1, 3, "height", "width"], [1, 3, None, 640], ["batch", 3, 640, 640]])
def test_model_input_size_rejects_models_exported_with_dynamic_shapes(shape):
    with pytest.raises(ValueError, match="fixed"):
        model_input_size(shape)


def fixed_output_model(path, input_size, box):
    """A stand-in ONNX model: a fixed [1, 3, height, width] input, and one constant detection out."""
    h, w = input_size
    out = helper.make_tensor("out", TensorProto.FLOAT, [1, 1, 6], [*box, 0.9, 0.0])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=out)], "fixed",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), path)
    return path


def test_the_detector_reads_its_input_size_from_the_model(tmp_path):
    detector = OnnxDetector(fixed_output_model(tmp_path / "m.onnx", (736, 1280), (0.0, 8.0, 10.0, 18.0)))

    assert detector.input_size == (736, 1280)


def test_the_detector_maps_boxes_from_a_rectangular_model_back_to_the_frame(tmp_path):
    # 1920x1080 into 1280x736: scale 2/3, 8 px of padding on top.
    detector = OnnxDetector(fixed_output_model(tmp_path / "m.onnx", (736, 1280), (100.0, 108.0, 300.0, 308.0)))

    dets = detector(np.zeros((1080, 1920, 3), dtype=np.uint8))

    assert [d.box for d in dets] == [pytest.approx((150.0, 150.0, 450.0, 450.0))]
