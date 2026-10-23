"""tether-model-compare --prep: timing the models with CUDA frame prep, the shipped configuration (#46)."""

import json

import pytest

from tether import model_compare
from tether.formats import save_timings
from tether.model_compare import main


def frames(n=5, frame_ms=5.0):
    return [{s: (i * frame_ms, (i + 1) * frame_ms) if s == "capture" else ((i + 1) * frame_ms,) * 2
             for s in ("capture", "prep", "detect", "track", "draw")} for i in range(n)]


def test_run_model_passes_the_frame_prep_to_tether_pipeline(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(model_compare, "cpp_binary", lambda cpp_dir, name: "tether-pipeline")
    monkeypatch.setattr(model_compare, "_run", lambda command, what: commands.append([str(c) for c in command]))

    model_compare.run_model(tmp_path, "clip.mp4", "m.onnx", "parallel", tmp_path / "t.jsonl", None, prep="cuda")
    model_compare.run_model(tmp_path, "clip.mp4", "m.onnx", "parallel", tmp_path / "t.jsonl", None)

    cuda, default = commands
    assert cuda[cuda.index("--prep") + 1] == "cuda"
    assert default[default.index("--prep") + 1] == "cpu"


def test_main_times_every_run_with_the_prep_asked_and_saves_it(tmp_path, monkeypatch):
    preps = []

    def fake_run(cpp_dir, video, model, mode, log, max_frames, prep="cpu"):
        preps.append(prep)
        save_timings(log, frames())

    monkeypatch.setattr(model_compare, "run_model", fake_run)
    monkeypatch.setattr(model_compare, "_input_size", lambda m: (736, 1280))
    out = tmp_path / "c.json"

    main(["clip.mp4", "--models", "a.onnx", "b.onnx", "--prep", "cuda", "--out", str(out)])

    assert preps == ["cuda"] * 4
    assert json.loads(out.read_text(encoding="utf-8"))["prep"] == "cuda"


def test_prep_is_cpu_or_cuda(tmp_path):
    with pytest.raises(SystemExit):
        main(["clip.mp4", "--models", "a.onnx", "--prep", "gpu", "--out", str(tmp_path / "c.json")])
    assert not (tmp_path / "c.json").exists()
