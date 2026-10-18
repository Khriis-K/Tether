from tether.demo import dump_detections
from tether.formats import Detection, load_detections


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
