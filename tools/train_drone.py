"""Fine-tune YOLO26n on the drone training set, and export a checkpoint to ONNX (#43, ADR 0004).

    pip install -e .[export]          # plus a CUDA build of PyTorch to train on the GPU
    tether-drone-dataset data/Anti-UAV300 data/antiuav-drone-stride10
    python tools/train_drone.py train data/antiuav-drone-stride10/data.yaml --epochs 1 --name pilot
    python tools/train_drone.py export RUN/weights/epoch12.pt models/yolo26n_drone_1280x736_e13.onnx   # epochN.pt is 0-based: epoch 13

Training starts from the COCO-pretrained yolo26n.pt with the settings in
train_args, fixed here (seed included) so a run can be repeated; Ultralytics
also saves them as args.yaml in the run folder. Ultralytics' AMP check
downloads yolo26n.pt into <current folder>/weights, so run it from a folder
out of git (e.g. models/) with absolute paths. Every epoch's checkpoint is
kept: the one shipped is chosen by AP50 on the session-disjoint val with
tether-drone-score, not by Ultralytics' own pick (best.pt), which is ranked on
a different metric over the stride-sampled val images.

Exports go through tools/export_model.py, with the shipped detector's settings: 1280x736,
batch 1, YOLO26's NMS-free head. The class names, {0: 'drone'}, go into the ONNX
file's metadata.
"""

import argparse
from pathlib import Path

PRETRAINED = Path(__file__).resolve().parent.parent / "models" / "yolo26n.pt"
CLASS_NAMES = {0: "drone"}
SEED = 0
BATCH = 8


def train_args(data: str, epochs: int, project: str, name: str) -> dict:
    """Ultralytics training settings; everything not set here is Ultralytics 8.4.181's default."""
    return {"data": data, "epochs": epochs, "project": project, "name": name,
            "imgsz": 1280,  # the export's width; training letterboxes to a 1280 square
            "batch": BATCH, "seed": SEED, "deterministic": True,
            # no single_cls: the dataset has one class already, and single_cls renames it "item"
            "save_period": 1,  # keep every epoch, to choose by val AP50 afterwards
            "workers": 4, "exist_ok": False, "plots": False}


def check_export_name(path: str | Path) -> Path:
    """Refuse a file name without "drone": its stem names the dump folder, so it mustn't match the COCO model's."""
    path = Path(path)
    if "drone" not in path.stem:
        raise argparse.ArgumentTypeError(f"{path}: a drone model's file name must contain 'drone', so it can't be "
                         "mistaken for the COCO model's")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fine-tune from the COCO-pretrained yolo26n.pt")
    train.add_argument("data", help="the dataset's data.yaml (from tether-drone-dataset)")
    train.add_argument("--pretrained", default=str(PRETRAINED), help=f"COCO-pretrained weights (default {PRETRAINED})")
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--project", default=str(PRETRAINED.parent / "drone-runs"),
                       help="folder for run folders (default models/drone-runs, out of git)")
    train.add_argument("--name", required=True, help="run folder name")
    export = commands.add_parser("export", help="export a checkpoint to ONNX at 1280x736")
    export.add_argument("weights", help="a checkpoint (.pt) from a training run")
    export.add_argument("out", type=check_export_name, help="the ONNX file to write; its name must contain 'drone'")
    args = parser.parse_args(argv)

    if args.command == "train":
        from ultralytics import YOLO  # imported here so the settings above are testable without Ultralytics

        YOLO(args.pretrained).train(**train_args(args.data, args.epochs, args.project, args.name))
        return
    from export_model import SHIPPED, export_onnx  # tools/ is on the path when this file is run

    # CLASS_NAMES whatever the checkpoint says, and the shipped detector's input size
    export_onnx(Path(args.weights), SHIPPED[1], args.out, names=CLASS_NAMES)
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
