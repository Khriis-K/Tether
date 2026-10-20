"""Writes the stand-in ONNX models the C++ tests load (needs the `onnx` package).

Each takes a fixed [1, 3, height, width] input and always outputs the same one
detection, [x1, y1, x2, y2, score, class] in model-input pixels, like YOLO26.
Run from the repo root: python cpp/tests/onnx/make_models.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).parent


def fixed_output_model(path: Path, input_size: tuple[int, int], row: tuple[float, ...]) -> None:
    h, w = input_size
    out = helper.make_tensor("out", TensorProto.FLOAT, [1, 1, 6], list(row))
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=out)], "fixed",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), path)


def dynamic_input_model(path: Path) -> None:
    out = helper.make_tensor("out", TensorProto.FLOAT, [1, 1, 6], [0.0, 0.0, 1.0, 1.0, 0.9, 0.0])
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=out)], "dynamic",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), path)


if __name__ == "__main__":
    # A 1920x1080 frame letterboxed to 1280x736 has scale 2/3 and 8 px of padding on top,
    # so this box maps back to (150, 150, 450, 450) in the frame.
    fixed_output_model(HERE / "fixed_736x1280.onnx", (736, 1280), (100.0, 108.0, 300.0, 308.0, 0.9, 0.0))
    dynamic_input_model(HERE / "dynamic_input.onnx")
