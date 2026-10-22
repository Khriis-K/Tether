# Running the benchmarks in Docker

A Linux image, built on NVIDIA's TensorRT container, that builds Tether and runs the video-file benchmarks with one command. It's for reproducing the numbers, not for the live demo: there's no webcam or window inside Docker.

## What you need

- An NVIDIA GPU and a driver recent enough for CUDA 13.4.
- Docker with the NVIDIA Container Toolkit, so `docker run --gpus all` works. On Windows, Docker Desktop with the WSL 2 backend provides it. Check with:
  ```
  docker run --rm --gpus all nvcr.io/nvidia/tensorrt:26.09-py3 nvidia-smi
  ```
- About 24 GB of disk: the built image is 23.6 GB, 19.4 GB of it NVIDIA's TensorRT image.

## Build

From the repo root:

```
docker build -f docker/Dockerfile -t tether .
```

The first build compiles OpenCV from source. The image is `nvcr.io/nvidia/tensorrt:26.09-py3`: TensorRT 11.3.0, CUDA 13.4.1, Ubuntu 24.04, Python 3.12.

CUDA frame prep is compiled for every GPU architecture the base image supports, compute capability 7.5 (Turing) to 12.0 (Blackwell, RTX 50xx), so the same image runs on any of them. To build faster for just your GPU, pass its architecture:

```
docker build -f docker/Dockerfile -t tether --build-arg CUDA_ARCHITECTURES=89 .
```

89 is Ada (RTX 40xx), 86 Ampere (RTX 30xx); `nvidia-smi --query-gpu=compute_cap --format=csv` gives yours (12.0 is 120).

## Datasets

Neither dataset can be redistributed, so download them yourself and mount them read-only at `/tether/data`, laid out as in the rest of the docs:

```
data/
  MOT17/train/MOT17-02-FRCNN/ ...            MOT17.zip from https://motchallenge.net/data/MOT17/
  Anti-UAV300/test/20190925_111757_1_1/ ...   Anti-UAV-RGBT.zip from https://github.com/ZhaoJ9014/Anti-UAV
```

Only MOT17's `train/` split is used. For Anti-UAV, only the `visible.*` files and `label_new/` are needed. The timing step uses the Anti-UAV test clip `20190925_111757_1_1/visible.mp4`.

## Models

Nothing to download by hand. On its first run, the benchmark exports YOLO26n at 1280×736 (the shipped detector, ADR 0003) and at 640 (MOT17's second row) with `tools/export_model.py`, which fetches the pretrained weights from Ultralytics, and builds the TensorRT engine for the shipped model with `trtexec`. Mount a folder at `/tether/models` to keep them between runs.

Use a separate folder from your host's `models/` (`models/docker/` above is one, and git ignores it). ONNX files would be fine to share, but a TensorRT engine only runs with the TensorRT version and GPU that built it, and the script reuses any engine it finds. An engine built by another TensorRT version (the Windows setup uses 11.4) fails to load.

## Run

```
docker run --rm --gpus all -v ./data:/tether/data:ro -v ./models/docker:/tether/models -v ./results/docker:/tether/results tether
```

That runs `docker/benchmark.sh`, which:

1. **Timing:** CPU against CUDA frame prep for the shipped configuration (TensorRT, parallel pipeline) on the Anti-UAV clip, two rounds each: `results/timing/cuda-prep-parallel.json`. As in [cuda-prep.md](benchmarks/cuda-prep.md). It goes first because it's the only GPU step, so a GPU problem shows up within minutes.
2. **MOT17 + ByteTrack:** Tether and ByteTrack on the same detections, for both models, plus the detector's own recall: `results/mot17/trackers/<name>/results.json`. As in [mot17-bytetrack.md](benchmarks/mot17-bytetrack.md).
3. **Anti-UAV dropout:** the dropout and gap tests on the test split, plus diagnostics: `results/antiuav/trackers/tether/results.json` and `results/antiuav/diagnose/`. As in [antiuav-dropout.md](benchmarks/antiuav-dropout.md).

Not covered: the other timing rows in `cuda-prep.md` (MOT17-04, sequential, 640) and the diagnostic rows in `mot17-bytetrack.md` (no low-score pass, recall at score ≥ 0.1). Each of their commands runs in the container as written, passed after `tether` in place of the default command.

On the dev laptop (RTX 5070 Laptop GPU, through WSL 2) the whole run took 45.7 minutes, and every accuracy number matched the write-ups exactly: [docker-reproduction.md](benchmarks/docker-reproduction.md).

Detections are dumped once under `results/mot17/detections/` and `results/antiuav/detections/` and reused by later runs, so start from an empty results folder to reproduce from scratch.

Any single step can be run on its own by overriding the command, for example:

```
docker run --rm --gpus all -v ./data:/tether/data:ro -v ./models/docker:/tether/models -v ./results/docker:/tether/results tether tether-mot17 data/MOT17
```

## Tests

The image builds the programs but not the C++ tests: building them runs `trtexec`, which needs the GPU that `docker build` doesn't get. Build and run them in a container:

```
docker run --rm --gpus all tether cpp/build.sh
```

And the Python tests:

```
docker run --rm tether pytest
```

## How it compares to the write-ups

The committed numbers come from Windows with TensorRT 11.4.0.106 on an RTX 5070 Laptop GPU. The image has TensorRT 11.3.0 on Linux, so its timings are a reproduction on a different setup, not the same measurement. On the same laptop the shipped configuration ran 256.72–259.71 FPS in the container against 195.45–195.86 on Windows, with detect, not decoding, the slowest stage ([docker-reproduction.md](benchmarks/docker-reproduction.md)).
