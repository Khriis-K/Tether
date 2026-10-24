"""tools/drone_breakdown.py: where a detector finds the drone, by recording session and by the class it named (#42)."""

import importlib.util
from pathlib import Path

from tether.antiuav import Clip
from tether.formats import Detection

_SPEC = importlib.util.spec_from_file_location(
    "drone_breakdown", Path(__file__).resolve().parents[2] / "tools" / "drone_breakdown.py")
drone_breakdown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drone_breakdown)

DRONE = (0.0, 0.0, 10.0, 10.0)
AWAY = (50.0, 50.0, 60.0, 60.0)


def det(box, score=0.9, cls=0):
    return Detection(box=box, score=score, cls=cls)


def test_recall_is_split_by_recording_session_over_drone_frames_only():
    a1 = Clip(name="20190925_100000_1_1", fps=20.0, boxes=[DRONE, DRONE, None])
    a2 = Clip(name="20190925_100000_1_2", fps=20.0, boxes=[DRONE])
    b = Clip(name="20190926_100000_1_1", fps=20.0, boxes=[DRONE, DRONE])
    frames = {a1.name: [[det(DRONE)], [], [det(AWAY)]], a2.name: [[det(DRONE)]], b.name: [[], [det(AWAY)]]}

    result = drone_breakdown.breakdown([(c, frames[c.name]) for c in (a1, a2, b)], threshold=0.25)

    assert result["by_session"] == {"20190925_100000": {"found": 2, "drones": 3, "recall": 2 / 3},
                                    "20190926_100000": {"found": 0, "drones": 2, "recall": 0.0}}


def test_each_found_drone_is_credited_to_the_class_of_its_best_scored_hit():
    clip = Clip(name="20190925_100000_1_1", fps=20.0, boxes=[DRONE, DRONE, DRONE])
    frames = [[det(DRONE, 0.5, cls=14), det(DRONE, 0.8, cls=4)], [det(DRONE, 0.6, cls=14)], [det(DRONE, 0.2, cls=4)]]

    result = drone_breakdown.breakdown([(clip, frames)], threshold=0.25)

    assert result["by_class"] == {"airplane": 1, "bird": 1}  # the 0.2 hit is under the threshold
