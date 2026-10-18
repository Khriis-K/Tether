"""The video source: frames from a webcam or a video file, one at a time.

Neither mode sleeps. A file is decoded as fast as the reader can go, so the
pipeline runs uncapped; a webcam read blocks until the camera's next frame, so
webcam mode follows the camera's rate.
"""

from collections.abc import Iterator

import cv2
import numpy as np


def frames(source: str) -> Iterator[np.ndarray]:
    """BGR frames, one at a time. `source` is a webcam index ("0") or a video file path.

    The source opens right away, not on the first frame, so opening a camera
    (which can take a second) isn't timed as the first frame's capture.
    """
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise OSError(f"can't open video source {source!r}")
    return _read(capture)


def _read(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()
