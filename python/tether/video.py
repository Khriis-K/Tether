"""The video source: frames from a webcam or a video file, one at a time.

Neither mode sleeps. A file is decoded as fast as the reader can go, so the
pipeline runs uncapped; a webcam read blocks until the camera's next frame, so
webcam mode follows the camera's rate.
"""

from collections.abc import Iterator

import cv2
import numpy as np


def frames(source: str) -> Iterator[np.ndarray]:
    """Yield BGR frames. `source` is a webcam index ("0") or a video file path."""
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise OSError(f"can't open video source {source!r}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()
