"""tools/detector_recall.py: detections scored as if each were its own one-frame track (#7)."""

import importlib.util
from pathlib import Path

from tether.formats import Detection

_SPEC = importlib.util.spec_from_file_location(
    "detector_recall", Path(__file__).resolve().parents[2] / "tools" / "detector_recall.py")
detector_recall = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(detector_recall)


def det(score, cls=0, x=0.0):
    return Detection(box=(x, 0.0, x + 10.0, 20.0), score=score, cls=cls)


def test_keeps_people_scored_at_least_the_cutoff():
    frames = [[det(0.9), det(0.25, x=20), det(0.24, x=40), det(0.9, cls=2, x=60)]]

    tracks = detector_recall.detections_as_tracks(frames, min_score=0.25)

    assert [t.box[0] for t in tracks[0]] == [0.0, 20.0]
    assert all(t.status == "tracked" and t.cls == 0 for t in tracks[0])


def test_every_detection_gets_its_own_id_so_no_identity_carries_over():
    frames = [[det(0.9), det(0.8, x=20)], [], [det(0.9)]]

    tracks = detector_recall.detections_as_tracks(frames, min_score=0.25)

    assert [[t.id for t in frame] for frame in tracks] == [[1, 2], [], [3]]
