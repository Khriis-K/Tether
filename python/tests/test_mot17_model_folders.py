"""tether-mot17 saves each model's detections under the model's file name (#21)."""

from test_mot17 import TETHER, GroundTruthDetector, mot17_dir  # noqa: F401 (mot17_dir is a fixture)
from tether.mot17 import evaluate, main


def test_main_reuses_the_detections_saved_under_the_model_file_name(tmp_path, mot17_dir):
    work = tmp_path / "work"
    evaluate(mot17_dir, work, TETHER, "tether", GroundTruthDetector(), detector_name="yolo26n_1280x736")
    # The model file doesn't exist, so running it would fail: passing means the saved detections were used.
    missing_model = tmp_path / "yolo26n_1280x736.onnx"

    assert main([str(mot17_dir), "--work-dir", str(work), "--model", str(missing_model)]) == 0
    assert [p.name for p in (work / "detections").iterdir()] == ["yolo26n_1280x736"]
