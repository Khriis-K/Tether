"""Detection and track files: the tracker's input and output, shared by every tool.

Both are JSON Lines, one line per frame, in frame order. See docs/FORMATS.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Point = tuple[float, float]


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


def save_detections(path: str | Path, frames: list[list[Detection]]) -> None:
    _write_lines(path, ({"frame": i, "detections": [_detection_to_json(d) for d in dets]}
                        for i, dets in enumerate(frames)))


def load_detections(path: str | Path) -> list[list[Detection]]:
    return [[_detection_from_json(d) for d in line["detections"]] for line in _read_lines(path)]


def save_tracks(path: str | Path, frames: list[list[Track]]) -> None:
    _write_lines(path, ({"frame": i, "tracks": [_track_to_json(t) for t in tracks]}
                        for i, tracks in enumerate(frames)))


def load_tracks(path: str | Path) -> list[list[Track]]:
    return [[_track_from_json(t) for t in line["tracks"]] for line in _read_lines(path)]


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


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def _box(values) -> Box:
    x1, y1, x2, y2 = (float(v) for v in values)
    return (x1, y1, x2, y2)


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
