"""Writes the stand-in ONNX models the C++ tests load (needs the `onnx` package).

Each takes a fixed [1, 3, height, width] input and always outputs the same one
detection, [x1, y1, x2, y2, score, class] in model-input pixels, like YOLO26.
Run from the repo root: python cpp/tests/onnx/make_models.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).parent


def fixed_output_model(path: Path, input_size: tuple[int, int], row: tuple[float, ...],
                       names: str | None = None) -> None:
    """`names`, when given, goes into the metadata as Ultralytics writes it, e.g. "{0: 'drone'}"."""
    h, w = input_size
    out = helper.make_tensor("out", TensorProto.FLOAT, [1, 1, 6], list(row))
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output0"], value=out)], "fixed",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    if names is not None:
        helper.set_model_props(model, {"names": names})
    onnx.save(model, path)


def brightness_model(path: Path, input_size: tuple[int, int], box: tuple[float, ...]) -> None:
    """One detection whose score is the input's mean value, so different frames give different
    outputs: a test can tell a fresh run from a replay of an old one."""
    h, w = input_size
    nodes = [
        helper.make_node("Constant", [], ["axes"], value=helper.make_tensor("a", TensorProto.INT64, [3], [1, 2, 3])),
        helper.make_node("ReduceMean", ["images", "axes"], ["mean"], keepdims=1),  # [1, 1, 1, 1]
        helper.make_node("Constant", [], ["shape"], value=helper.make_tensor("s", TensorProto.INT64, [3], [1, 1, 1])),
        helper.make_node("Reshape", ["mean", "shape"], ["score"]),
        helper.make_node("Constant", [], ["box"], value=helper.make_tensor("b", TensorProto.FLOAT, [1, 1, 4], list(box))),
        helper.make_node("Constant", [], ["cls"], value=helper.make_tensor("c", TensorProto.FLOAT, [1, 1, 1], [0.0])),
        helper.make_node("Concat", ["box", "score", "cls"], ["output0"], axis=2),
    ]
    graph = helper.make_graph(
        nodes, "brightness",
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
    # The same, carrying class names in its metadata like the drone model (fixed_736x1280 has none).
    fixed_output_model(HERE / "drone_736x1280.onnx", (736, 1280), (100.0, 108.0, 300.0, 308.0, 0.9, 0.0),
                       names="{0: 'drone'}")
    dynamic_input_model(HERE / "dynamic_input.onnx")
    # Same frame geometry; the score is the letterboxed frame's mean value (0 to 1).
    brightness_model(HERE / "brightness_736x1280.onnx", (736, 1280), (100.0, 108.0, 300.0, 308.0))
