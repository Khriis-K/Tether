"""What to call each class ID. Mirrors cpp/src/tether/class_names.hpp.

A model exported by Ultralytics names its classes in its ONNX metadata
("names": "{0: 'drone'}"). COCO's names are the fallback, for models without
any; a flag (comma-separated names) overrides both.
"""

import ast
from pathlib import Path

COCO_NAMES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush")


def parse_names_metadata(text: str) -> tuple[str, ...]:
    """Ultralytics' "names" metadata, a Python dict literal, as names by class ID.

    Raises ValueError unless it's a dict of names whose IDs are 0, 1, ... in order.
    """
    try:
        body = ast.parse(text, mode="eval").body
    except SyntaxError:
        body = None
    # Read from the parse tree, not literal_eval's dict, which would quietly keep the last of a repeated ID.
    entries = ([(k.value, v.value) for k, v in zip(body.keys, body.values)
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)]
               if isinstance(body, ast.Dict) else [])
    if (not entries or len(entries) != len(body.keys) or [k for k, _ in entries] != list(range(len(entries)))
            or not all(isinstance(v, str) for _, v in entries)):
        raise ValueError(f"class names metadata {text!r} isn't {{0: 'name', ...}}")
    return tuple(v for _, v in entries)


def parse_names_flag(text: str) -> tuple[str, ...]:
    """--class-names: comma-separated names by class ID ("drone" or "drone,bird")."""
    names = tuple(text.split(","))
    if not all(names):
        raise ValueError(f"--class-names has an empty name: {text!r}")
    return names


def model_class_names(model: str | Path) -> tuple[str, ...] | None:
    """The names in `model`'s ONNX metadata; None when it has none, the file isn't there, or it isn't
    an .onnx file (a TensorRT engine has no metadata). Raises ValueError, naming the model, for bad names."""
    if Path(model).suffix.lower() != ".onnx" or not Path(model).exists():
        return None
    import onnxruntime as ort  # imported here: the scoring tools that use this don't otherwise need it

    names = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"]).get_modelmeta() \
        .custom_metadata_map.get("names")
    try:
        return None if names is None else parse_names_metadata(names)
    except ValueError as e:
        raise ValueError(f"{model}: {e}") from None


def class_names_for(model: str | Path, flag: str | None) -> tuple[str, ...]:
    """The flag's names if given, else the model's, else COCO's."""
    if flag is not None:
        return parse_names_flag(flag)
    return model_class_names(model) or COCO_NAMES


def class_name(cls: int, names: tuple[str, ...]) -> str:
    """The name of class ID `cls`; "class N" for an ID `names` doesn't have."""
    return names[cls] if 0 <= cls < len(names) else f"class {cls}"
