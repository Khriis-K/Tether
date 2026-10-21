"""Detection, track and timing files, shared by every tool.

Detection and track files are the tracker's input and output; timing logs hold
how long each pipeline step took. All are JSON Lines, one line per frame, in
frame order. See docs/FORMATS.md.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Point = tuple[float, float]

STEPS = ("capture", "prep", "detect", "track", "draw")  # pipeline steps, in order
StepTimes = dict[str, tuple[float, float]]  # step -> (start, end) in milliseconds


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    cls: int


@dataclass(frozen=True)
class Track:
    id: int
    box: Box
    cls: int
    status: str  # "tentative", "tracked" or "searching"
    predicted_box: Box
    frames_since_seen: int
    path: tuple[Point, ...]  # recent box centers, oldest first


def save_detections(path: str | Path, frames: Iterable[list[Detection]]) -> None:
    """Write frames as they come, so a stream (like a webcam) can be saved without buffering."""
    _write_lines(path, ({"frame": i, "detections": [_detection_to_json(d) for d in dets]}
                        for i, dets in enumerate(frames)))


def load_detections(path: str | Path) -> list[list[Detection]]:
    return [[_detection_from_json(d) for d in line["detections"]] for line in _read_lines(path)]


def save_tracks(path: str | Path, frames: list[list[Track]]) -> None:
    _write_lines(path, ({"frame": i, "tracks": [_track_to_json(t) for t in tracks]}
                        for i, tracks in enumerate(frames)))


def load_tracks(path: str | Path) -> list[list[Track]]:
    return [[_track_from_json(t) for t in line["tracks"]] for line in _read_lines(path)]


def save_timings(path: str | Path, frames: Iterable[StepTimes]) -> None:
    """Write frames as they come, so a run cut short still leaves a valid log."""
    _write_lines(path, ({"frame": i, "steps": {step: _floats(times[step]) for step in STEPS}}
                        for i, times in enumerate(frames)))


def load_timings(path: str | Path) -> list[StepTimes]:
    return [_step_times_from_json(path, line) for line in _read_lines(path)]


def load_dropped(path: str | Path) -> int | None:
    """Frames the camera dropped over a webcam run, from each line's optional "dropped" count.

    None for a run that didn't count them (a video file, or the Python pipeline).
    """
    counts = [line["dropped"] for line in _read_lines(path) if "dropped" in line]
    return sum(counts) if counts else None


def _detection_to_json(d: Detection) -> dict:
    return {"box": _floats(d.box), "score": float(d.score), "class": int(d.cls)}


def _detection_from_json(d: dict) -> Detection:
    return Detection(box=_box(d["box"]), score=float(d["score"]), cls=int(d["class"]))


def _track_to_json(t: Track) -> dict:
    return {
        "id": int(t.id),
        "box": _floats(t.box),
        "class": int(t.cls),
        "status": t.status,
        "predicted_box": _floats(t.predicted_box),
        "frames_since_seen": int(t.frames_since_seen),
        "path": [_floats(p) for p in t.path],
    }


def _track_from_json(t: dict) -> Track:
    return Track(
        id=int(t["id"]),
        box=_box(t["box"]),
        cls=int(t["class"]),
        status=t["status"],
        predicted_box=_box(t["predicted_box"]),
        frames_since_seen=int(t["frames_since_seen"]),
        path=tuple((float(x), float(y)) for x, y in t["path"]),
    )


def _step_times_from_json(path: str | Path, line: dict) -> StepTimes:
    missing = [step for step in STEPS if step not in line["steps"]]
    if missing:
        raise ValueError(f"{path}: frame {line['frame']} is missing steps {missing}")
    return {step: _span(line["steps"][step]) for step in STEPS}


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _box(values) -> Box:
    x1, y1, x2, y2 = (float(v) for v in values)
    return (x1, y1, x2, y2)


def _span(values) -> tuple[float, float]:
    start, end = (float(v) for v in values)
    return (start, end)


def _write_lines(path: str | Path, objects) -> None:
    # newline="\n" so the bytes are the same on Windows and Linux.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")


def _read_lines(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for i, line in enumerate(lines):
        if line["frame"] != i:
            raise ValueError(f"{path}: expected frame {i}, found frame {line['frame']}")
    return lines
