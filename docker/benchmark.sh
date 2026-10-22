#!/bin/sh
# The one-command benchmark (#19), run by the Docker image: exports the models
# and builds the shipped model's TensorRT engine if they're missing, then times
# the shipped configuration, scores MOT17 against ByteTrack and runs the Anti-UAV
# dropout test, writing everything under results/. The GPU steps (the engine
# build, then timing) go first, so a missing GPU or a stale engine fails in
# minutes, not hours.
# Expects data/MOT17 and data/Anti-UAV300 (docs/DOCKER.md).
set -e
cd "$(dirname "$0")/.."

SHIPPED=models/yolo26n_1280x736  # ADR 0003
OLD_DEFAULT=models/yolo26n       # YOLO26n at 640, MOT17's second row
CLIP=data/Anti-UAV300/test/20190925_111757_1_1/visible.mp4  # the timing clip of #14-#18

[ -f "$SHIPPED.onnx" ] || python tools/export_model.py
[ -f "$OLD_DEFAULT.onnx" ] || python tools/export_model.py --size 640
# An engine only runs on the GPU and TensorRT version that built it, so don't
# reuse one built outside this image.
[ -f "$SHIPPED.engine" ] || trtexec --onnx="$SHIPPED.onnx" --saveEngine="$SHIPPED.engine" --noTF32 --skipInference

# Timing: CPU vs CUDA frame prep on the shipped model, parallel (docs/benchmarks/cuda-prep.md).
tether-runtime-compare "$CLIP" --compare frame-prep --pipeline parallel --model "$SHIPPED.onnx" \
    --out results/timing/cuda-prep-parallel.json

# MOT17 + ByteTrack on the same detections (docs/benchmarks/mot17-bytetrack.md).
for model in "$SHIPPED" "$OLD_DEFAULT"; do
    stem=$(basename "$model")
    tether-mot17 data/MOT17 --model "$model.onnx" --name "tether_$stem"
    tether-mot17 data/MOT17 --model "$model.onnx" --name "bytetrack_$stem" --tracker tether-bytetrack
    python tools/detector_recall.py data/MOT17 "results/mot17/detections/$stem"
done

# Anti-UAV dropout (docs/benchmarks/antiuav-dropout.md).
tether-antiuav data/Anti-UAV300
for condition in drop-0.0-seed0 drop-0.2-seed0; do
    python tools/antiuav_diagnose.py data/Anti-UAV300 "results/antiuav/detections/$condition" \
        "results/antiuav/trackers/tether/$condition"
done

echo "done: results/timing, results/mot17/trackers, results/antiuav"
