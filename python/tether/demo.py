"""tether-demo: run YOLO26 + the tracker on a webcam or video file and show the tracks.

With --dump, skips the tracker and window and writes the raw detections to a
detection file instead, for benchmarks and other trackers to read.
"""

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from tether import video
from tether.detector import DETECTOR_MIN_SCORE, OnnxDetector
from tether.draw import draw_tracks
from tether.formats import Detection, save_detections
from tether.tracker import Tracker

DEFAULT_MODEL = Path("models/yolo26n.onnx")
WINDOW = "Tether"
QUIT_KEYS = (ord("q"), 27)  # q or Esc


def dump_detections(frames: Iterable[np.ndarray], detect: Callable[[np.ndarray], list[Detection]],
                    path: str | Path) -> None:
    """Write every frame's detections, as they come, without tracking or filtering."""
    save_detections(path, (detect(frame) for frame in frames))


def show_tracks(frames: Iterable[np.ndarray], detect: Callable[[np.ndarray], list[Detection]],
                tracker: Tracker) -> None:
    """Track and display frames until the source ends or the window is closed."""
    for frame in frames:
        draw_tracks(frame, tracker.update(detect(frame)))
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF in QUIT_KEYS or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tether-demo", description=__doc__)
    parser.add_argument("source", help='webcam index (e.g. "0") or video file path')
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help=f"YOLO26 ONNX model (default {DEFAULT_MODEL}; see tools/export_model.py)")
    parser.add_argument("--dump", metavar="DETECTIONS",
                        help="write a detection file (.jsonl) instead of tracking and showing")
    parser.add_argument("--det-min-score", type=float, default=DETECTOR_MIN_SCORE,
                        help="lowest detector score kept; the tracker then applies its own "
                             f"threshold on top (default {DETECTOR_MIN_SCORE})")
    args = parser.parse_args(argv)

    detect = OnnxDetector(args.model, min_score=args.det_min_score)
    try:
        if args.dump:
            dump_detections(video.frames(args.source), detect, args.dump)
        else:
            show_tracks(video.frames(args.source), detect, Tracker())
    except KeyboardInterrupt:
        pass  # Ctrl-C ends a webcam run; frames written so far are a valid detection file
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
