"""Download pretrained YOLO26n and export it to ONNX at models/yolo26n.onnx.

    pip install -e .[export]
    python tools/export_model.py

Repeatable: the Ultralytics version is pinned in pyproject.toml, and the
export settings are fixed here. nms=False asks for YOLO26's NMS-free head, so
the model outputs final boxes as [1, 300, 6] rows of x1, y1, x2, y2, score, class.
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

MODELS = Path(__file__).resolve().parent.parent / "models"
NAME = "yolo26n"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    weights = Path(attempt_download_asset(MODELS / f"{NAME}.pt"))  # skipped if already there
    onnx_path = YOLO(str(weights)).export(format="onnx", imgsz=640, batch=1, dynamic=False, nms=False,
                                       opset=18,
                                       simplify=False)  # simplifying would pip-install onnxslim mid-export
    print(f"exported {onnx_path}")


if __name__ == "__main__":
    main()
