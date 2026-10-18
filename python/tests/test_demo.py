from tether.demo import dump_detections, timed_tracking
from tether.formats import STEPS, Detection, load_detections
from tether.tracker import Tracker


class FakeClock:
    """Milliseconds that pass only when a fake step says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def spend(self, ms: float) -> None:
        self.now += ms


def test_tracking_logs_every_step_of_every_frame_in_order():
    clock = FakeClock()
    det = Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=0)
    seen = []

    def frames():
        for name in ("f0", "f1"):
            clock.spend(2)
            yield name

    def prepare(frame):
        clock.spend(1)
        return frame + "-prepared"

    def detect(prepared):
        seen.append(prepared)
        clock.spend(10)
        return [det]

    def show(frame, tracks):
        seen.append((frame, [t.box for t in tracks]))
        clock.spend(3)
        return True

    log = list(timed_tracking(frames(), prepare, detect, Tracker(), show, clock))

    assert seen == ["f0-prepared", ("f0", [det.box]), "f1-prepared", ("f1", [det.box])]
    assert [list(steps) for steps in log] == [list(STEPS)] * 2
    assert log[1] == {"capture": (16.0, 18.0), "prep": (18.0, 19.0), "detect": (19.0, 29.0),
                      "track": (29.0, 29.0), "draw": (29.0, 32.0)}


def test_tracking_stops_after_the_frame_where_the_viewer_quits():
    shown = []

    def show(frame, tracks):
        shown.append(frame)
        return frame != "f1"

    log = list(timed_tracking(["f0", "f1", "f2"], lambda f: f, lambda p: [], Tracker(), show, FakeClock()))

    assert shown == ["f0", "f1"]
    assert len(log) == 2


def test_dump_writes_one_line_per_frame_with_every_detection_unfiltered(tmp_path):
    per_frame = {
        0: [Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.95, cls=0),
            Detection(box=(20.0, 20.0, 30.0, 30.0), score=0.11, cls=2)],
        1: [],
        2: [Detection(box=(5.0, 5.0, 15.0, 15.0), score=0.12, cls=0)],
    }
    frames = ["frame0", "frame1", "frame2"]
    path = tmp_path / "dets.jsonl"

    dump_detections(frames, lambda frame: per_frame[frames.index(frame)], path)

    assert load_detections(path) == [per_frame[0], per_frame[1], per_frame[2]]


def test_dump_of_an_empty_source_writes_an_empty_file(tmp_path):
    path = tmp_path / "dets.jsonl"

    dump_detections([], lambda frame: [], path)

    assert path.read_bytes() == b""
