"""Settings and export naming for tools/train_drone.py (#43)."""

import argparse
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "train_drone", Path(__file__).resolve().parents[2] / "tools" / "train_drone.py")
train_drone = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_drone)


def test_training_is_seeded_and_deterministic_so_it_can_be_repeated():
    args = train_drone.train_args("data.yaml", epochs=3, project="runs", name="pilot")
    assert args["seed"] == 0 and args["deterministic"] is True


def test_training_runs_at_the_shipped_input_width():
    args = train_drone.train_args("data.yaml", epochs=3, project="runs", name="pilot")
    assert args["imgsz"] == 1280


def test_training_keeps_the_datasets_class_name():
    # single_cls would rename the one class "item", and the export's metadata would say so
    args = train_drone.train_args("data.yaml", epochs=3, project="runs", name="pilot")
    assert not args.get("single_cls")


def test_the_exported_class_is_drone():
    assert train_drone.CLASS_NAMES == {0: "drone"}


def test_every_epochs_checkpoint_is_kept_so_it_can_be_chosen_on_val_ap50():
    args = train_drone.train_args("data.yaml", epochs=3, project="runs", name="pilot")
    assert args["save_period"] == 1


def test_the_epochs_and_run_folder_are_the_given_ones():
    args = train_drone.train_args("d/data.yaml", epochs=7, project="C:/runs", name="full")
    assert (args["data"], args["epochs"], args["project"], args["name"]) == ("d/data.yaml", 7, "C:/runs", "full")


@pytest.mark.parametrize("name", ["yolo26n_1280x736.onnx", "yolo26n.onnx", "models/yolo26n_1280x736.onnx"])
def test_an_export_without_drone_in_its_name_is_refused(name):
    # The dump folder and the scorer's model field are the file's stem, so a drone model named
    # like the COCO one would land in the COCO baseline's dump.
    with pytest.raises(argparse.ArgumentTypeError):  # argparse shows its message as the error
        train_drone.check_export_name(name)


def test_an_export_named_drone_is_accepted():
    assert train_drone.check_export_name("models/yolo26n_drone_1280x736_e12.onnx") == Path(
        "models/yolo26n_drone_1280x736_e12.onnx")
