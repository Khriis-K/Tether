"""Class names: from a flag, else the model's ONNX metadata, else COCO's (#46). Mirrors cpp/tests/test_class_names.cpp."""

import importlib.util
import json
from pathlib import Path

import pytest

from tether.antiuav import Clip
from tether.class_names import COCO_NAMES, class_name, class_names_for, model_class_names, parse_names_flag, \
    parse_names_metadata
from tether.formats import Detection

ONNX = Path(__file__).resolve().parents[2] / "cpp" / "tests" / "onnx"  # stand-ins from make_models.py
DRONE_MODEL = ONNX / "drone_736x1280.onnx"  # names {0: 'drone'}
NAMELESS_MODEL = ONNX / "fixed_736x1280.onnx"  # no metadata

_SPEC = importlib.util.spec_from_file_location(
    "drone_breakdown", Path(__file__).resolve().parents[2] / "tools" / "drone_breakdown.py")
drone_breakdown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drone_breakdown)


def test_coco_names_are_the_eighty_yolo26_classes_in_order():
    assert len(COCO_NAMES) == 80
    assert (COCO_NAMES[0], COCO_NAMES[4], COCO_NAMES[79]) == ("person", "airplane", "toothbrush")


def test_metadata_is_ultralytics_dict_of_ids_to_names():
    assert parse_names_metadata("{0: 'drone'}") == ("drone",)
    assert parse_names_metadata("{0: 'person', 1: 'traffic light', 2: \"it's\"}") == ("person", "traffic light",
                                                                                      "it's")


@pytest.mark.parametrize("bad", ["", "{}", "drone", "{0: drone}", "{0: 'drone'", "{1: 'drone'}", "{0: 'a', 2: 'b'}",
                                 "{0: 'a', 0: 'b'}", "{0: 1}", "{0: 'a', 1: b}", "['drone']"])
def test_metadata_that_isnt_ids_zero_to_n_is_refused(bad):
    with pytest.raises(ValueError):
        parse_names_metadata(bad)


def test_the_flag_is_comma_separated_names_by_class_id():
    assert parse_names_flag("drone") == ("drone",)
    assert parse_names_flag("drone,bird") == ("drone", "bird")
    for bad in ("", "drone,", ",drone", "a,,b"):
        with pytest.raises(ValueError):
            parse_names_flag(bad)


def test_model_names_come_from_the_onnx_metadata():
    assert model_class_names(DRONE_MODEL) == ("drone",)


def test_model_names_are_none_without_metadata_or_without_the_file(tmp_path):
    assert model_class_names(NAMELESS_MODEL) is None
    assert model_class_names(tmp_path / "gone.onnx") is None


def test_model_names_are_none_for_an_engine_given_as_the_model(tmp_path):
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"not an onnx file")
    assert model_class_names(engine) is None


def test_unreadable_metadata_names_the_model(tmp_path):
    with pytest.raises(ValueError, match="bad_names.onnx"):
        model_class_names(_model_with_names(tmp_path / "bad_names.onnx", "{1: 'drone'}"))


def _model_with_names(path, names):
    import onnx
    model = onnx.load(str(NAMELESS_MODEL))
    onnx.helper.set_model_props(model, {"names": names})
    onnx.save(model, str(path))
    return path


def test_names_take_the_flag_then_the_model_then_coco():
    assert class_names_for(DRONE_MODEL, "uav") == ("uav",)
    assert class_names_for(DRONE_MODEL, None) == ("drone",)
    assert class_names_for(NAMELESS_MODEL, None) == COCO_NAMES


def test_an_id_past_the_names_is_called_class_n():
    assert class_name(0, ("drone",)) == "drone"
    assert class_name(1, ("drone",)) == "class 1"
    assert class_name(-1, COCO_NAMES) == "class -1"


DRONE = (0.0, 0.0, 10.0, 10.0)


def test_breakdown_credits_class_zero_of_the_drone_model_to_drone():
    clip = Clip(name="20190925_100000_1_1", fps=20.0, boxes=[DRONE])

    result = drone_breakdown.breakdown([(clip, [[Detection(box=DRONE, score=0.9, cls=0)]])], threshold=0.25,
                                       names=("drone",))

    assert result["by_class"] == {"drone": 1}


def test_breakdown_reads_names_from_the_model_its_dump_names(tmp_path):
    (tmp_path / "dump.json").write_text(json.dumps({"model_path": str(DRONE_MODEL)}), encoding="utf-8")
    assert drone_breakdown.dump_class_names(tmp_path, None) == ("drone",)
    assert drone_breakdown.dump_class_names(tmp_path, "uav") == ("uav",)


def test_breakdown_falls_back_to_coco_for_a_dump_without_dump_json(tmp_path):
    assert drone_breakdown.dump_class_names(tmp_path, None) == COCO_NAMES


def test_breakdown_refuses_a_dump_whose_model_is_gone_unless_given_names(tmp_path):
    (tmp_path / "dump.json").write_text(json.dumps({"model_path": str(tmp_path / "gone.onnx")}), encoding="utf-8")
    with pytest.raises(SystemExit, match="--class-names"):
        drone_breakdown.dump_class_names(tmp_path, None)
    assert drone_breakdown.dump_class_names(tmp_path, "drone") == ("drone",)
