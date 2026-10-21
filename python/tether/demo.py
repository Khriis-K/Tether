"""tether-demo: run YOLO26 + the tracker on a webcam or video file and show the tracks.

Every tracking run logs each frame's per-step timing to a timing log (see
--timing); tether-timing summarizes it. --dump runs aren't timed.

With --headless, tracks and logs timing without drawing or a window, as the
C++ pipeline (tether-pipeline) does, so the two can be compared step for step.

With --dump, skips the tracker and window and writes the raw detections to a
detection file instead, for benchmarks and other trackers to read.
"""

import argparse
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from tether import video
from tether.detector import DETECTOR_MIN_SCORE, OnnxDetector, PreparedFrame, prepare
from tether.draw import draw_tracks
from tether.formats import Detection, StepTimes, Track, save_detections, save_timings
from tether.tracker import Tracker

DEFAULT_MODEL = Path("models/yolo26n_1280x736.onnx")  # the shipped detector (ADR 0003), shared by every tool
WINDOW = "Tether"
QUIT_KEYS = (ord("q"), 27)  # q or Esc


def dump_detections(frames: Iterable[np.ndarray], detect: Callable[[np.ndarray], list[Detection]],
                    path: str | Path) -> None:
    """Write every frame's detections, as they come, without tracking or filtering."""
    save_detections(path, (detect(frame) for frame in frames))


def timed_tracking(frames: Iterable[np.ndarray], prepare: Callable[[np.ndarray], PreparedFrame],
                   detect: Callable[[PreparedFrame], list[Detection]], tracker: Tracker,
                   show: Callable[[np.ndarray, list[Track]], bool],
                   clock: Callable[[], float]) -> Iterator[StepTimes]:
    """Run each frame through every step, yielding the frame's step times as it finishes.

    Stops when the source ends or `show` returns False (that frame is still yielded).
    `clock` reads milliseconds. `step` records into `times`, which is a fresh dict each frame.
    """
    @contextmanager
    def step(name):
        start = clock()
        yield
        times[name] = (start, clock())

    frames = iter(frames)
    while True:
        times = {}
        with step("capture"):
            frame = next(frames, None)
        if frame is None:
            return
        with step("prep"):
            prepared = prepare(frame)
        with step("detect"):
            detections = detect(prepared)
        with step("track"):
            tracks = tracker.update(detections)
        with step("draw"):
            keep_going = show(frame, tracks)
        yield times
        if not keep_going:
            return


def show_frame(frame: np.ndarray, tracks: list[Track]) -> bool:
    """Draw and display one frame. False once the viewer quits or closes the window."""
    draw_tracks(frame, tracks)
    cv2.imshow(WINDOW, frame)
    quit_pressed = cv2.waitKey(1) & 0xFF in QUIT_KEYS
    return not quit_pressed and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def headless(frame: np.ndarray, tracks: list[Track]) -> bool:
    """The draw step with nothing to draw on: never stops the run."""
    return True


def clock_ms() -> float:
    return time.perf_counter() * 1000


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
    parser.add_argument("--headless", action="store_true",
                        help="track and log timing without drawing or showing a window")
    parser.add_argument("--timing", metavar="LOG",
                        help="timing log to write (.jsonl; default results/timing-<date>-<time>.jsonl)")
    args = parser.parse_args(argv)

    detect = OnnxDetector(args.model, min_score=args.det_min_score)
    try:
        if args.dump:
            dump_detections(video.frames(args.source), detect, args.dump)
        else:
            timing_log = Path(args.timing or f"results/timing-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
            timing_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"logging timing to {timing_log}")
            prepare_frame = partial(prepare, input_size=detect.input_size)
            save_timings(timing_log, timed_tracking(video.frames(args.source), prepare_frame, detect.detect,
                                                Tracker(), headless if args.headless else show_frame, clock_ms))
    except KeyboardInterrupt:
        pass  # Ctrl-C ends a webcam run; frames written so far make a valid detection file or timing log
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
