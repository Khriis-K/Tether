"""The detector: one video frame in, that frame's detections out.

Runs a YOLO26 model exported to ONNX (see tools/export_model.py) with ONNX
Runtime. YOLO26 has no NMS step: the model itself outputs at most 300 final
boxes per image as rows of [x1, y1, x2, y2, score, class], in model-input
pixels. All that's left to do here is undo the letterbox and drop low scores.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from tether.formats import Detection

INPUT_SIZE = 640  # YOLO26 is exported for 640x640 input
DETECTOR_MIN_SCORE = 0.1  # low on purpose: detection files are saved unfiltered
PAD_VALUE = 114  # gray, the same padding Ultralytics trains with


def letterbox(frame: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize to fit a size x size square, keeping the aspect ratio, and pad the rest.

    Returns the padded image, the scale applied, and the (x, y) padding before the image.
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    image = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    resized = frame if (new_w, new_h) == (w, h) else cv2.resize(frame, (new_w, new_h),
                                                                interpolation=cv2.INTER_LINEAR)
    image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return image, scale, (pad_x, pad_y)


def decode(output: np.ndarray, scale: float, pad: tuple[int, int],
           frame_shape: tuple[int, int], min_score: float) -> list[Detection]:
    """Turn the model's [1, N, 6] output into detections in frame pixels.

    Boxes are clipped to the frame; any left with no area are dropped, so every
    box has x2 > x1 and y2 > y1 as the detection file requires.
    """
    h, w = frame_shape
    pad_x, pad_y = pad
    detections = []
    for x1, y1, x2, y2, score, cls in output[0]:
        if score < min_score:
            continue
        x1 = min(max((x1 - pad_x) / scale, 0.0), w)
        x2 = min(max((x2 - pad_x) / scale, 0.0), w)
        y1 = min(max((y1 - pad_y) / scale, 0.0), h)
        y2 = min(max((y2 - pad_y) / scale, 0.0), h)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(box=(float(x1), float(y1), float(x2), float(y2)),
                                    score=float(score), cls=int(cls)))
    return detections


class OnnxDetector:
    """YOLO26 on ONNX Runtime's CPU provider. Call it with a BGR frame (as OpenCV reads them).

    The provider is fixed so timings mean the same thing on every machine;
    faster runtimes come later as their own setting.
    """

    def __init__(self, model_path: str | Path, min_score: float = DETECTOR_MIN_SCORE) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._min_score = min_score

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        image, scale, pad = letterbox(frame)
        # BGR HWC uint8 -> RGB CHW float in [0, 1], with a batch dimension.
        blob = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        (output,) = self._session.run(None, {self._input_name: blob})
        return decode(output, scale, pad, frame.shape[:2], self._min_score)
