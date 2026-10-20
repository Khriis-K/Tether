import random

from tether.cli import main
from tether.formats import Detection, load_tracks, save_detections


def scripted_scene(n_frames=40, seed=0):
    """Two objects moving steadily, plus jittered noise boxes, for a busy input."""
    rng = random.Random(seed)
    frames = []
    for f in range(n_frames):
        dets = [
            Detection(box=(2.0 * f, 50.0, 2.0 * f + 20.0, 70.0), score=0.9, cls=0),
            Detection(box=(300.0 - 3.0 * f, 200.0, 330.0 - 3.0 * f, 240.0), score=0.8, cls=2),
        ]
        for _ in range(rng.randint(0, 3)):
            x, y = rng.uniform(0, 600), rng.uniform(0, 400)
            dets.append(Detection(box=(x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40)),
                                  score=rng.uniform(0.1, 0.6), cls=rng.randint(0, 2)))
        rng.shuffle(dets)
        frames.append(dets)
    return frames


def test_cli_writes_one_track_frame_per_detection_frame(tmp_path):
    dets_path = tmp_path / "dets.jsonl"
    tracks_path = tmp_path / "tracks.jsonl"
    save_detections(dets_path, [[Detection(box=(0.0, 0.0, 10.0, 10.0), score=0.9, cls=0)], []])

    assert main([str(dets_path), str(tracks_path), "--tentative-misses", "0"]) == 0

    frames = load_tracks(tracks_path)
    assert len(frames) == 2
    assert len(frames[0]) == 1
    assert frames[1] == []


def test_same_input_run_twice_gives_byte_identical_output(tmp_path):
    dets_path = tmp_path / "dets.jsonl"
    save_detections(dets_path, scripted_scene())
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"

    main([str(dets_path), str(out_a)])
    main([str(dets_path), str(out_b)])

    assert out_a.read_bytes() == out_b.read_bytes()
    assert len(out_a.read_bytes()) > 0
