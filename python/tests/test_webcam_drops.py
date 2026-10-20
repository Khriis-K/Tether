"""Frames a webcam dropped, as the C++ pipeline records them in its timing log (#11)."""

import json

from tether.formats import load_dropped, save_timings
from tether.timing import main

FRAME = {"capture": (0.0, 2.0), "prep": (2.0, 3.0), "detect": (3.0, 13.0), "track": (13.0, 14.0), "draw": (14.0, 16.0)}


def webcam_log(path, dropped):
    """A timing log as a webcam run writes it: every line also says how many frames were dropped before it."""
    lines = []
    for i, n in enumerate(dropped):
        start = 20.0 * i
        steps = {step: [start + a, start + b] for step, (a, b) in FRAME.items()}
        lines.append(json.dumps({"frame": i, "steps": steps, "dropped": n}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dropped_frames_add_up_over_the_run(tmp_path):
    assert load_dropped(webcam_log(tmp_path / "t.jsonl", [0, 3, 0, 1])) == 4


def test_a_file_run_has_no_dropped_count(tmp_path):
    log = tmp_path / "t.jsonl"
    save_timings(log, [FRAME, FRAME])

    assert load_dropped(log) is None


def test_the_summary_of_a_webcam_run_includes_its_dropped_frames(tmp_path, capsys):
    log = webcam_log(tmp_path / "t.jsonl", [0, 2, 0])
    out = tmp_path / "s.json"

    main([str(log), "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["dropped_frames"] == 2
    assert "2 dropped" in capsys.readouterr().out
