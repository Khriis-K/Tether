"""Download a pretrained YOLO26 model and export it to ONNX in models/.

    pip install -e .[export]
    python tools/export_model.py                             # models/yolo26n.onnx, the shipped 640x640 model
    python tools/export_model.py --model n --size 1280x736   # models/yolo26n_1280x736.onnx
    python tools/export_model.py --model s --size 640        # models/yolo26s_640x640.onnx

The size is width x height (one number means a square) and must be a multiple
of 32, the model's stride. Every export other than the shipped one is named
after its model and size, so it can't overwrite models/yolo26n.onnx, and the
detector reads the size back from the model.

Repeatable: the Ultralytics version is pinned in pyproject.toml, and the
export settings are fixed here. nms=False asks for YOLO26's NMS-free head, so
the model outputs final boxes as [1, 300, 6] rows of x1, y1, x2, y2, score, class.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
MODEL_VARIANTS = ("n", "s", "m")
SHIPPED = ("n", (640, 640))  # model variant, (height, width)
STRIDE = 32


def parse_size(text: str) -> tuple[int, int]:
    """'1280x736' (width x height) or '640' (a square) -> (height, width)."""
    try:
        w, h = (int(v) for v in text.split("x")) if "x" in text else (int(text),) * 2
    except ValueError:
        raise argparse.ArgumentTypeError(f"size {text!r} must be WIDTHxHEIGHT or one number") from None
    if w <= 0 or h <= 0 or w % STRIDE or h % STRIDE:
        # Ultralytics would round it up to a multiple of 32 itself, and the file name would lie.
        raise argparse.ArgumentTypeError(f"size {text!r} must be positive multiples of {STRIDE}")
    return h, w


def output_name(model: str, size: tuple[int, int]) -> str:
    """yolo26n.onnx for the shipped model, else e.g. yolo26n_1280x736.onnx (width x height)."""
    if (model, size) == SHIPPED:
        return f"yolo26{model}.onnx"
    h, w = size
    return f"yolo26{model}_{w}x{h}.onnx"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a pretrained YOLO26 model to ONNX in models/.")
    parser.add_argument("--model", choices=MODEL_VARIANTS, default=SHIPPED[0], help="model variant (default n)")
    parser.add_argument("--size", type=parse_size, default=SHIPPED[1],
                        help="input size as WIDTHxHEIGHT, or one number for a square (default 640)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from ultralytics import YOLO  # imported here so the naming above is testable without Ultralytics
    from ultralytics.utils.downloads import attempt_download_asset

    args = parse_args(argv)
    MODELS.mkdir(exist_ok=True)
    weights = Path(attempt_download_asset(MODELS / f"yolo26{args.model}.pt"))  # skipped if already there
    target = MODELS / output_name(args.model, args.size)
    # Ultralytics writes <weights-stem>.onnx next to the weights, which for yolo26n.pt is the
    # shipped model. Exporting a copy in a scratch folder keeps models/ untouched until the move.
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(shutil.copy(weights, scratch))
        onnx_path = YOLO(str(copy)).export(format="onnx", imgsz=list(args.size), batch=1, dynamic=False,
                                           nms=False, opset=18,
                                           simplify=False)  # simplifying would pip-install onnxslim mid-export
        shutil.move(onnx_path, target)
    print(f"exported {target}")


if __name__ == "__main__":
    main()
