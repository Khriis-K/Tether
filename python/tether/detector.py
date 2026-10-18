"""The detector: one video frame in, that frame's detections out.

Runs a YOLO26 model exported to ONNX (see tools/export_model.py) with ONNX
Runtime. YOLO26 has no NMS step: the model itself outputs at most 300 final
boxes per image as rows of [x1, y1, x2, y2, score, class], in model-input
pixels. All that's left to do here is undo the letterbox and drop low scores.

The input size isn't set here: each model is exported at one fixed height x
width, and the detector reads it from the model and letterboxes to it.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from tether.formats import Detection

DETECTOR_MIN_SCORE = 0.1  # low on purpose: detection files are saved unfiltered
PAD_VALUE = 114  # gray, the same padding Ultralytics trains with


def letterbox(frame: np.ndarray, size: int | tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize to fit a (height, width) input, keeping the aspect ratio, and pad the rest.

    A single number means a square. Returns the padded image, the scale applied,
    and the (x, y) padding before the image.
    """
    out_h, out_w = (size, size) if isinstance(size, int) else size
    h, w = frame.shape[:2]
    scale = min(out_h / h, out_w / w)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (out_w - new_w) // 2, (out_h - new_h) // 2
    image = np.full((out_h, out_w, 3), PAD_VALUE, dtype=np.uint8)
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


@dataclass(frozen=True)
class PreparedFrame:
    """Model input for one frame, plus what's needed to map boxes back to the frame."""
    blob: np.ndarray  # [1, 3, height, width] float32 RGB in [0, 1]
    scale: float
    pad: tuple[int, int]
    frame_shape: tuple[int, int]  # (h, w) of the original frame


def prepare(frame: np.ndarray, input_size: tuple[int, int]) -> PreparedFrame:
    """Frame prep: letterbox a BGR frame (as OpenCV reads them) into a (height, width) model input."""
    image, scale, pad = letterbox(frame, input_size)
    # BGR HWC uint8 -> RGB CHW float in [0, 1], with a batch dimension.
    blob = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    return PreparedFrame(blob, scale, pad, frame.shape[:2])


def model_input_size(shape: list) -> tuple[int, int]:
    """(height, width) from an ONNX input shape [1, 3, height, width].

    Dynamic dimensions come back from ONNX Runtime as names or None; this
    detector needs a model exported at a fixed size, so those are an error.
    """
    if not all(isinstance(d, int) for d in shape):
        raise ValueError(f"model input shape {shape} must be fixed; export with dynamic=False")
    return shape[2], shape[3]


class OnnxDetector:
    """YOLO26 on ONNX Runtime's CPU provider. Call it with a BGR frame (as OpenCV reads them),
    or call `detect` with a frame already prepared for `input_size`.

    The provider is fixed so timings mean the same thing on every machine;
    faster runtimes come later as their own setting.
    """

    def __init__(self, model_path: str | Path, min_score: float = DETECTOR_MIN_SCORE) -> None:
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self.input_size = model_input_size(model_input.shape)  # (height, width)
        self._min_score = min_score

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(prepare(frame, self.input_size))

    def detect(self, prepared: PreparedFrame) -> list[Detection]:
        (output,) = self._session.run(None, {self._input_name: prepared.blob})
        return decode(output, prepared.scale, prepared.pad, prepared.frame_shape, self._min_score)
