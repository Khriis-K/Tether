"""tether-pipeline-compare against the real C++ build (#11). Run by ctest, not by a plain pytest run.

ctest passes the build folder in TETHER_CPP_DIR. The file name doesn't start with
test_, so pytest only collects it when named on the command line.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from tether.pipeline_compare import main

STAND_IN_MODEL = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "onnx" / "fixed_736x1280.onnx"


def test_both_real_pipelines_run_on_the_same_frames(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.zeros((1080, 1920, 3), dtype=np.uint8))
    out = tmp_path / "compare.json"

    assert main([str(tmp_path / "f%d.png"), "--model", str(STAND_IN_MODEL),
                 "--cpp-dir", os.environ["TETHER_CPP_DIR"], "--out", str(out)]) == 0

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["python"]["frames"] == result["cpp"]["frames"] == 3
