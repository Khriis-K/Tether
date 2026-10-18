"""Basic display: a box and an ID for each confirmed track, drawn onto the frame."""

import colorsys

import cv2
import numpy as np

from tether.formats import Track

GOLDEN_RATIO = 0.618033988749895  # steps hues apart so neighboring IDs never look alike


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    """Draw in place. Tentative tracks are skipped: they may be one-frame noise.

    They still use up IDs, so the IDs on screen can have gaps. A searching track
    is drawn thin at its last seen box, marked "?". (Its predicted_box is the
    guess for the next frame, so drawing that here would run a frame ahead.)
    """
    for track in tracks:
        if track.status == "tentative":
            continue
        searching = track.status == "searching"
        x1, y1, x2, y2 = (round(v) for v in track.box)
        color = track_color(track.id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1 if searching else 2)
        label = f"{track.id}?" if searching else str(track.id)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def track_color(track_id: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((track_id * GOLDEN_RATIO) % 1.0, 0.85, 1.0)
    return round(b * 255), round(g * 255), round(r * 255)  # OpenCV wants BGR
