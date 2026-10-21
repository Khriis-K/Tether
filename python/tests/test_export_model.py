"""Naming and argument parsing for tools/export_model.py (#21)."""

import argparse
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "export_model", Path(__file__).resolve().parents[2] / "tools" / "export_model.py")
export_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_model)


def test_size_is_read_as_width_x_height_and_returned_as_height_width():
    assert export_model.parse_size("1280x736") == (736, 1280)


def test_a_single_number_means_a_square():
    assert export_model.parse_size("1280") == (1280, 1280)


@pytest.mark.parametrize("text", ["1280x720", "650", "0", "-640", "big", "1280x"])
def test_sizes_that_are_not_positive_multiples_of_32_are_rejected(text):
    # Ultralytics would quietly round 1280x720 up to 1280x736, so the file name would lie.
    with pytest.raises(argparse.ArgumentTypeError):
        export_model.parse_size(text)


def test_the_output_name_includes_model_and_input_size():
    assert export_model.output_name("n", (736, 1280)) == "yolo26n_1280x736.onnx"
    assert export_model.output_name("s", (640, 640)) == "yolo26s_640x640.onnx"
    assert export_model.output_name("n", (1280, 1280)) == "yolo26n_1280x1280.onnx"


def test_the_default_export_is_the_shipped_model():
    assert export_model.output_name(*export_model.SHIPPED) == "yolo26n_1280x736.onnx"


def test_the_first_shipped_model_keeps_its_old_name():
    # YOLO26n at 640 shipped until ADR 0003; earlier results and commands name it yolo26n.onnx.
    assert export_model.output_name("n", (640, 640)) == "yolo26n.onnx"


def test_arguments_default_to_the_shipped_model():
    args = export_model.parse_args([])

    assert (args.model, args.size) == ("n", (736, 1280))


def test_arguments_take_model_and_size():
    args = export_model.parse_args(["--model", "m", "--size", "1280x736"])

    assert (args.model, args.size) == ("m", (736, 1280))


def test_only_known_model_sizes_are_accepted():
    with pytest.raises(SystemExit):
        export_model.parse_args(["--model", "q"])
