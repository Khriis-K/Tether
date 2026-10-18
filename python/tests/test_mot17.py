import json
import sys

import cv2
import numpy as np
import pytest

from tether.formats import Detection, Track
from tether.mot17 import PERSON, evaluate, save_mot_results, score


def track(id, box, status="tracked", cls=PERSON):
    return Track(id=id, box=box, cls=cls, status=status, predicted_box=box,
                 frames_since_seen=0 if status != "searching" else 1, path=())


# --- MOT results writer ---

def test_writes_tracked_people_as_one_based_frames_with_top_left_width_height(tmp_path):
    path = tmp_path / "seq.txt"

    save_mot_results(path, [[track(3, (10.0, 20.0, 30.0, 60.0))], [], [track(3, (11.5, 20.0, 31.5, 60.0))]],
                     cls=PERSON)

    assert path.read_text(encoding="utf-8").splitlines() == [
        "1,3,10.00,20.00,20.00,40.00,1,-1,-1,-1",
        "3,3,11.50,20.00,20.00,40.00,1,-1,-1,-1",
    ]


def test_leaves_out_tentative_and_searching_tracks_and_other_classes(tmp_path):
    path = tmp_path / "seq.txt"
    box = (0.0, 0.0, 10.0, 10.0)

    save_mot_results(path, [[track(1, box, status="tentative"), track(2, box, status="searching"),
                             track(3, box, cls=2), track(4, box)]], cls=PERSON)

    assert path.read_text(encoding="utf-8").splitlines() == ["1,4,0.00,0.00,10.00,10.00,1,-1,-1,-1"]


def test_no_tracks_gives_an_empty_file(tmp_path):
    path = tmp_path / "seq.txt"

    save_mot_results(path, [[], []], cls=PERSON)

    assert path.read_bytes() == b""


# --- Scoring with TrackEval, on a tiny fake MOT17 ---

# Two people walking right for 10 frames, as (id, x, y, w, h) per frame.
N_FRAMES = 10
PEOPLE = {1: (0.0, 0.0), 2: (200.0, 0.0)}


def person_box(start, frame):
    x, y = start
    return (x + 5.0 * frame, y, x + 5.0 * frame + 40.0, y + 100.0)


def make_sequence(root, name, with_images=False):
    seq = root / name
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={N_FRAMES}\n"
        "imWidth=320\nimHeight=240\nimExt=.jpg\n", encoding="utf-8")
    rows = []
    for f in range(N_FRAMES):
        for pid, start in PEOPLE.items():
            x1, y1, x2, y2 = person_box(start, f)
            rows.append(f"{f + 1},{pid},{x1},{y1},{x2 - x1},{y2 - y1},1,1,1.0")
    (seq / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if with_images:
        (seq / "img1").mkdir()
        for f in range(N_FRAMES):
            cv2.imwrite(str(seq / "img1" / f"{f + 1:06d}.jpg"), np.zeros((240, 320, 3), np.uint8))
    return seq


def write_tracker(trackers_dir, tracker, seq_name, frames):
    data = trackers_dir / tracker / "data"
    data.mkdir(parents=True, exist_ok=True)
    save_mot_results(data / f"{seq_name}.txt", frames, cls=PERSON)


def perfect_frames(swap_at=None):
    """Tracks equal to the ground truth. From `swap_at` on, the two IDs trade places."""
    frames = []
    for f in range(N_FRAMES):
        ids = {1: 1, 2: 2} if swap_at is None or f < swap_at else {1: 2, 2: 1}
        frames.append(sorted((track(ids[pid], person_box(start, f)) for pid, start in PEOPLE.items()),
                             key=lambda t: t.id))
    return frames


@pytest.fixture
def gt_dir(tmp_path):
    gt = tmp_path / "gt"
    make_sequence(gt, "SEQ-A")
    make_sequence(gt, "SEQ-B")
    return gt


def test_perfect_tracks_score_full_marks(tmp_path, gt_dir):
    trackers = tmp_path / "trackers"
    write_tracker(trackers, "t", "SEQ-A", perfect_frames())
    write_tracker(trackers, "t", "SEQ-B", perfect_frames())

    results = score(gt_dir, trackers, "t", ["SEQ-A", "SEQ-B"])

    for scores in (results["sequences"]["SEQ-A"], results["sequences"]["SEQ-B"], results["combined"]):
        assert scores["HOTA"] == pytest.approx(100.0)
        assert scores["IDF1"] == pytest.approx(100.0)
        assert scores["IDSW"] == 0


def test_id_swaps_are_counted_per_sequence_and_summed(tmp_path, gt_dir):
    trackers = tmp_path / "trackers"
    write_tracker(trackers, "t", "SEQ-A", perfect_frames(swap_at=5))
    write_tracker(trackers, "t", "SEQ-B", perfect_frames())

    results = score(gt_dir, trackers, "t", ["SEQ-A", "SEQ-B"])

    assert results["sequences"]["SEQ-A"]["IDSW"] == 2  # both people switch ID once
    assert results["sequences"]["SEQ-A"]["IDF1"] == pytest.approx(50.0)
    assert results["sequences"]["SEQ-B"]["IDSW"] == 0
    assert results["combined"]["IDSW"] == 2


def test_a_tracker_that_outputs_nothing_scores_zero(tmp_path, gt_dir):
    trackers = tmp_path / "trackers"
    write_tracker(trackers, "t", "SEQ-A", [[] for _ in range(N_FRAMES)])
    write_tracker(trackers, "t", "SEQ-B", [[] for _ in range(N_FRAMES)])

    results = score(gt_dir, trackers, "t", ["SEQ-A", "SEQ-B"])

    assert results["combined"]["HOTA"] == pytest.approx(0.0)
    assert results["combined"]["IDF1"] == pytest.approx(0.0)


# --- The whole evaluation, one call ---

class GroundTruthDetector:
    """Detects every ground-truth person, plus one non-person box, frame by frame.

    Counts frames across calls, so it must see each sequence's frames in order.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        f = self.calls % N_FRAMES
        self.calls += 1
        people = [Detection(box=person_box(start, f), score=0.9, cls=PERSON) for start in PEOPLE.values()]
        return people + [Detection(box=(100.0, 150.0, 130.0, 180.0), score=0.9, cls=2)]


@pytest.fixture
def mot17_dir(tmp_path):
    root = tmp_path / "MOT17"
    make_sequence(root / "train", "MOT17-02-FRCNN", with_images=True)
    make_sequence(root / "train", "MOT17-04-FRCNN", with_images=True)
    make_sequence(root / "train", "MOT17-04-DPM", with_images=True)  # same video, other public detector
    return root


TETHER = f'"{sys.executable}" -m tether.cli'


def test_evaluates_every_frcnn_sequence_and_saves_the_results(tmp_path, mot17_dir):
    work = tmp_path / "work"

    results = evaluate(mot17_dir, work, TETHER, "tether", GroundTruthDetector())

    assert sorted(results["sequences"]) == ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]
    # Tether confirms a track on its 3rd frame, so frames 1-2 are missed: 8 of 10 frames found.
    # Boxes match exactly, so HOTA = sqrt(DetA * AssA) = sqrt(8/10 * 8/10).
    for scores in (*results["sequences"].values(), results["combined"]):
        assert scores["IDSW"] == 0
        assert scores["IDF1"] == pytest.approx(100 * 2 * 8 / (10 + 8))
        assert scores["HOTA"] == pytest.approx(80.0)
    saved = json.loads((work / "trackers" / "tether" / "results.json").read_text(encoding="utf-8"))
    assert saved == results
    assert saved["tracker"] == "tether"


def test_reruns_reuse_the_saved_detections(tmp_path, mot17_dir):
    work = tmp_path / "work"
    evaluate(mot17_dir, work, TETHER, "tether", GroundTruthDetector())
    detector = GroundTruthDetector()

    evaluate(mot17_dir, work, TETHER, "tether", detector)

    assert detector.calls == 0


def test_any_tracker_command_can_be_scored_the_same_way(tmp_path, mot17_dir):
    # A stand-in tracker that reads a detection file and reports no tracks at all.
    script = tmp_path / "empty_tracker.py"
    script.write_text(
        "import sys\n"
        "lines = open(sys.argv[1], encoding='utf-8').read().splitlines()\n"
        "with open(sys.argv[2], 'w', encoding='utf-8', newline='\\n') as f:\n"
        "    for i, _ in enumerate(lines):\n"
        "        f.write('{\"frame\": %d, \"tracks\": []}\\n' % i)\n", encoding="utf-8")

    results = evaluate(mot17_dir, tmp_path / "work", f'"{sys.executable}" "{script}"', "empty",
                       GroundTruthDetector())

    assert results["tracker"] == "empty"
    assert results["combined"]["HOTA"] == pytest.approx(0.0)
    assert (tmp_path / "work" / "trackers" / "empty" / "results.json").exists()


def test_a_tracker_that_drops_frames_is_refused(tmp_path, mot17_dir):
    # Writes only frame 0, so every later frame would be misnumbered or lost.
    script = tmp_path / "short_tracker.py"
    script.write_text(
        "import sys\n"
        "with open(sys.argv[2], 'w', encoding='utf-8', newline='\\n') as f:\n"
        "    f.write('{\"frame\": 0, \"tracks\": []}\\n')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1 frames"):
        evaluate(mot17_dir, tmp_path / "work", f'"{sys.executable}" "{script}"', "short",
                 GroundTruthDetector())


def test_a_sequence_with_missing_images_is_refused_and_leaves_no_detections(tmp_path, mot17_dir):
    (mot17_dir / "train" / "MOT17-02-FRCNN" / "img1" / "000004.jpg").unlink()

    with pytest.raises(ValueError, match="seqinfo.ini says 10"):
        evaluate(mot17_dir, tmp_path / "work", TETHER, "tether", GroundTruthDetector())

    assert list((tmp_path / "work" / "detections" / "yolo26n").iterdir()) == []


def test_each_detector_gets_its_own_saved_detections(tmp_path, mot17_dir):
    work = tmp_path / "work"
    evaluate(mot17_dir, work, TETHER, "tether", GroundTruthDetector(), detector_name="yolo26n")
    detector = GroundTruthDetector()

    evaluate(mot17_dir, work, TETHER, "tether", detector, detector_name="yolo26s")

    assert detector.calls == 2 * N_FRAMES  # both sequences detected afresh


def test_results_name_the_detection_set(tmp_path, mot17_dir):
    results = evaluate(mot17_dir, tmp_path / "work", TETHER, "tether", GroundTruthDetector(),
                       detector_name="yolo26n_1280x736")

    assert results["detections"] == "yolo26n_1280x736"


def test_recall_and_precision_ignore_boxes_on_static_people(tmp_path, gt_dir):
    # A static person (class 7) stands at (100, 150) in SEQ-A. The official rule drops
    # boxes that match one, rather than counting them as false alarms.
    static = (100.0, 150.0, 130.0, 210.0)
    with open(gt_dir / "SEQ-A" / "gt" / "gt.txt", "a", encoding="utf-8") as f:
        f.writelines(f"{i + 1},3,100,150,30,60,0,7,1.0\n" for i in range(N_FRAMES))
    found_one = [[track(1, person_box(PEOPLE[1], f)), track(2, static)]
                 + ([track(3, (300.0, 0.0, 320.0, 50.0))] if f < 5 else [])  # a false alarm in 5 frames
                 for f in range(N_FRAMES)]
    trackers = tmp_path / "trackers"
    write_tracker(trackers, "t", "SEQ-A", found_one)

    results = score(gt_dir, trackers, "t", ["SEQ-A"])

    assert results["sequences"]["SEQ-A"]["Recall"] == pytest.approx(100 * 10 / 20)
    assert results["sequences"]["SEQ-A"]["Precision"] == pytest.approx(100 * 10 / 15)
