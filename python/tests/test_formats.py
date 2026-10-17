from tether.formats import (
    Detection,
    Track,
    load_detections,
    load_tracks,
    save_detections,
    save_tracks,
)


def test_detections_round_trip_including_empty_frames(tmp_path):
    frames = [
        [Detection(box=(10.0, 20.0, 30.0, 40.0), score=0.9, cls=0)],
        [],
        [
            Detection(box=(1.5, 2.5, 3.5, 4.5), score=0.12, cls=2),
            Detection(box=(5.0, 6.0, 7.0, 8.0), score=0.5, cls=0),
        ],
    ]
    path = tmp_path / "dets.jsonl"

    save_detections(path, frames)

    assert load_detections(path) == frames


def test_tracks_round_trip_including_empty_frames(tmp_path):
    frames = [
        [
            Track(
                id=1,
                box=(10.0, 20.0, 30.0, 40.0),
                cls=0,
                status="tracked",
                predicted_box=(11.0, 21.0, 31.0, 41.0),
                frames_since_seen=0,
                path=((20.0, 30.0), (21.0, 31.0)),
            )
        ],
        [],
    ]
    path = tmp_path / "tracks.jsonl"

    save_tracks(path, frames)

    assert load_tracks(path) == frames


def test_files_use_lf_line_endings_on_every_platform(tmp_path):
    dets_path = tmp_path / "dets.jsonl"
    tracks_path = tmp_path / "tracks.jsonl"

    save_detections(dets_path, [[], []])
    save_tracks(tracks_path, [[], []])

    assert dets_path.read_bytes() == b'{"frame": 0, "detections": []}\n{"frame": 1, "detections": []}\n'
    assert tracks_path.read_bytes() == b'{"frame": 0, "tracks": []}\n{"frame": 1, "tracks": []}\n'


def test_detection_file_is_one_json_line_per_frame(tmp_path):
    path = tmp_path / "dets.jsonl"

    save_detections(path, [[Detection(box=(1.0, 2.0, 3.0, 4.0), score=0.5, cls=7)], []])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        '{"frame": 0, "detections": [{"box": [1.0, 2.0, 3.0, 4.0], "score": 0.5, "class": 7}]}',
        '{"frame": 1, "detections": []}',
    ]
