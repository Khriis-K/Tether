# Tether

Real-time multi-object tracker in C++ that keeps stable IDs on fast-moving and temporarily blocked objects.

> 🚧 **Work in progress.** So far the Python tracker runs on detection files and live in a basic demo, and the C++ tracker runs on detection files with output that matches it. Benchmark numbers will be added here once they're real.

## What it will do

- **Detect automatically.** A pretrained YOLO26n detector at 1280×736 input finds the objects in each frame, so nothing has to be selected by hand.
- **Keep IDs stable.** A hand-written tracker (Kalman filter + Hungarian matching) predicts motion, so objects keep their IDs through fast movement and short blockages.
- **Run fast.** A C++ pipeline on an NVIDIA GPU: ONNX model, TensorRT, custom CUDA frame prep, and parallel stages.
- **Show its work.** A live HUD with boxes, IDs, trails, predicted positions, "searching" markers, and FPS / worst-case frame time / end-to-end delay.
- **Prove it.** Accuracy on MOT17 against ByteTrack, robustness on Anti-UAV drone footage, and before-vs-after speed numbers for every upgrade.

## Status

Built in stages: a Python prototype first, then a C++ port, then benchmarks, then GPU speed work. Progress is tracked in the [milestones](https://github.com/Khriis-K/Tether/milestones).

**Drone detection (stage 5):** the fine-tuned drone model `yolo26n_drone_1280x736` was scored once on held-out data ([docs/benchmarks/drone-test.md](docs/benchmarks/drone-test.md)). What it can claim:
- On new flights from the Anti-UAV site's one camera, it finds the drone on 88.4% of frames at 97.5% precision. The COCO model it started from (`yolo26n_1280x736`, any class counted as a drone) gets 22.2% and 75.7% at the tracker's 0.25, and 45.7% and 44.8% at its own best cut.
- It meets the speed targets in the shipped configuration, measured on one laptop GPU (RTX 5070 Laptop) on one val clip.

What it can't claim yet:
- **ID retention.** Tracked end to end, it keeps the drone's ID on 75.3% of test frames, short of its target of 95% of what the labeled boxes give (82.6%).
- **False alarms.** It calls a bird a drone on 19 of 86 bird images. On 2.25 min of this site's empty sky it makes 190 false detections, 174 of them on one lamp in a night clip.
- **Transfer.** On the Seraphim drone images it scores below the untrained COCO model (AP50 0.164 against 0.216).

It isn't a counter-UAS system.

## Python prototype

```
pip install -e .[test,demo]
tether-track detections.jsonl tracks.jsonl
pytest
```

### Live demo

The demo runs pretrained YOLO26n through ONNX Runtime. Export the model once:

```
pip install -e .[demo,export]
python tools/export_model.py
```

That's `models/yolo26n_1280x736.onnx`, the shipped detector and the default of every tool that takes a model ([ADR 0003](docs/adr/0003-ship-yolo26n-1280x736.md)). On ONNX Runtime's CPU provider it's slow: detect takes 2.20–2.29× as long as at 640 (72.68–76.26 against 32.77–33.33 ms, `docs/benchmarks/tensorrt-runtime.md`). For a CPU-only machine or a live webcam, export the 640 model too (`python tools/export_model.py --size 640`, giving `models/yolo26n.onnx`) and pass `--model models/yolo26n.onnx`.

Then point it at a webcam index or a video file. Press `q` or Esc to stop.

```
tether-demo 0
tether-demo clip.mp4
```

`--dump detections.jsonl` skips the tracker and saves every detection down to `--det-min-score` (default 0.1). The tracker applies its own threshold (`--min-score` on `tether-track`, default 0.25), so other trackers can be run on the same file.

The detection and track file formats are in [docs/FORMATS.md](docs/FORMATS.md).

### MOT17 evaluation

Download and unzip [MOT17](https://motchallenge.net/data/MOT17/) (`MOT17.zip`), then:

```
pip install -e .[demo,eval]
tether-mot17 path/to/MOT17
```

This scores the tracker on the 7 MOT17 train videos with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, IDF1, ID switches) and saves per-video and combined results to `results/mot17/trackers/tether/results.json`. Detections are dumped once to `results/mot17/detections/<model>/` and reused, so every tracker sees the same input; `--model` picks another model, which gets its own folder. Any tracker that reads a detection file and writes a track file can be scored with `--tracker "<command>" --name <name>`; it's run as `<command> DETECTIONS TRACKS`.

The ByteTrack baseline (Ultralytics' BYTETracker) is scored on the same detections:

```
pip install -e .[bytetrack]
tether-mot17 path/to/MOT17 --tracker tether-bytetrack --name bytetrack_yolo26n_1280x736
```

Its tests (`python/tests/test_bytetrack.py`) are skipped unless the `bytetrack` extra is installed.

`python tools/detector_recall.py path/to/MOT17 results/mot17/detections/<model>` scores the detections alone, the ceiling for any tracker. Results so far: [docs/benchmarks/mot17-bytetrack.md](docs/benchmarks/mot17-bytetrack.md).

## C++ tracker

The same tracker in C++ (`cpp/`), reading and writing the same files. It needs CMake 3.20+ and a C++20 compiler (MSVC 2022 on Windows; GCC 13 on Linux also builds it). The first configure downloads the pinned test framework and JSON library.

```
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release
```

`cpp/build.sh` runs all three. On Windows, Visual Studio 2022 ships CMake under `Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin` if it isn't on your PATH. The CLI takes the same options as the Python one, except the diagnostic `--single-target`:

```
cpp/build/Release/tether-track detections.jsonl tracks.jsonl --confirm-frames 3
```

On the 7 MOT17 videos and 40 Anti-UAV clips under two dropout conditions, its IDs and statuses match the Python tracker's on every frame: [docs/benchmarks/cpp-port-parity.md](docs/benchmarks/cpp-port-parity.md).

### TensorRT runtime

`tether-pipeline --runtime tensorrt` runs the model on the GPU through TensorRT instead of on ONNX Runtime's CPU provider (`--runtime onnxruntime`, the default). It's optional: TensorRT's download needs an NVIDIA login, so CMake doesn't fetch it, and without it everything else builds as before. Unpack a TensorRT release built for your CUDA version (tested: TensorRT 11.4.0.106 for CUDA 13.4) and point CMake at it, and at that CUDA toolkit if more than one is installed:

```
cmake -S cpp -B cpp/build -DTensorRT_ROOT=C:/path/to/TensorRT-11.4.0.106 -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"
```

The same build compiles CUDA frame prep (below) with nvcc, for the RTX 5070's architecture (`CMAKE_CUDA_ARCHITECTURES`, default 120). With the Visual Studio generator, also pick the toolkit for nvcc with `-T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"`, in a fresh build folder.

Then build the engine from the ONNX model, once per machine, with the `trtexec` that ships in TensorRT's `bin` folder (67–90 s for YOLO26n on the laptop it was tested on):

```
trtexec --onnx=models/yolo26n_1280x736.onnx --saveEngine=models/yolo26n_1280x736.engine --noTF32 --skipInference
```

The shipped configuration, which the spec's speed targets are for, is that engine with CUDA frame prep and the parallel pipeline: 195.45–195.86 FPS, P99 frame time 6.49–6.57 ms on the Anti-UAV clip (`docs/benchmarks/cuda-prep.md`). The runtime, prep and pipeline defaults stay ONNX Runtime, CPU and sequential, so `tether-pipeline VIDEO` alone still works without a GPU. The shipped configuration is:

```
tether-pipeline VIDEO --runtime tensorrt --prep cuda --pipeline parallel
```

`--runtime tensorrt` loads the engine next to the model: `models/yolo26n_1280x736.onnx` (the default, also `--model-size n`; `s` and `m` pick `yolo26s_1280x736` and `yolo26m_1280x736`) loads `models/yolo26n_1280x736.engine`. An engine only runs on the GPU model and TensorRT version that built it, and only at the input size it was built for, so engines stay out of git (`*.engine`, `*.plan`) and each model and input size needs its own (rebuild it when the ONNX file changes too; the pipeline doesn't check): `models/yolo26n.onnx` (640) becomes `models/yolo26n.engine` the same way. TensorRT 11 takes its precision from the model, float32 here. `--noTF32` keeps it plain float32: `trtexec` otherwise allows TF32, which moved 0.66–1.05% of scores on MOT17-04 by more than the agreement tolerance allows, while saving at most 0.341 ms a frame (see the write-up).

TensorRT replays the engine from a CUDA graph captured at start-up, one launch per frame instead of one per kernel (2.10–2.24× faster detect at 640, `docs/benchmarks/cuda-graph.md`); `--no-cuda-graph` turns that off.

`--prep cuda` (with `--runtime tensorrt`) prepares each frame on the GPU instead of the CPU: one CUDA kernel letterboxes, reorders and scales it, and TensorRT reads the result where it is, with no trip through the CPU. It gives the same input as CPU prep (bit-identical on 1920×1080 frames). On the Anti-UAV clip with YOLO26n at 1280×736, prep is 9.19–9.21× faster and parallel FPS goes from 126.63–127.71 to 195.45–195.86 (`docs/benchmarks/cuda-prep.md`). `--prep cpu` is the default, and the only choice for ONNX Runtime.

`tether-runtime-compare VIDEO --model models/yolo26n.onnx` checks the two runtimes against each other on one video: their detections must agree within a set tolerance, and it times both. Results: [docs/benchmarks/tensorrt-runtime.md](docs/benchmarks/tensorrt-runtime.md). With `--compare cuda-graph` it does the same for TensorRT without and with the graph, `--compare frame-prep` for CPU against CUDA frame prep, and `--pipeline parallel` times parallel runs.

`tether-model-compare VIDEO --models M1.onnx M2.onnx ...` times several models on TensorRT on the same video (parallel by default, the model order rotating each round), names each run's slowest stage, marks each against the spec's speed targets, and with `--accuracy-dir results/model-size/trackers` adds each model's MOT17 scores from `tether-mot17`. The YOLO26 n/s/m × 640/1280×736 table and the recommendation for the shipped detector: [docs/benchmarks/model-size.md](docs/benchmarks/model-size.md).

### Drone model

`models/yolo26n_drone_1280x736.onnx` is YOLO26n at 1280×736 fine-tuned on Anti-UAV to find one class, drone ([ADR 0004](docs/adr/0004-drone-detector-on-anti-uav.md), [docs/benchmarks/drone-model.md](docs/benchmarks/drone-model.md)). It's a separate model: the COCO model stays the default. Its weights stay out of git, so train and export it with `tools/train_drone.py` (see its docstring), build its engine like the COCO one, and pass it with `--model`:

```
trtexec --onnx=models/yolo26n_drone_1280x736.onnx --saveEngine=models/yolo26n_drone_1280x736.engine --noTF32 --skipInference
tether-pipeline VIDEO --model models/yolo26n_drone_1280x736.onnx --runtime tensorrt --prep cuda --pipeline parallel --show
```

The stats panel names each track's class from the model's ONNX metadata, so the drone model's tracks read "drone" and the COCO model's read COCO's names, as before. On TensorRT the metadata is read from the `.onnx` file beside the engine; with only the engine, pass `--class-names drone`. Its speed in the shipped configuration: [docs/benchmarks/drone-speed.md](docs/benchmarks/drone-speed.md). The tracker inside `tether-pipeline` uses its default score cutoff, 0.25. The drone model's val-tuned cutoff, 0.352, is a `tether-track --min-score` setting for scoring dumped detections ([docs/benchmarks/drone-tracking.md](docs/benchmarks/drone-tracking.md)). Its one test run, against the targets set before it (two pass, four miss): [docs/benchmarks/drone-test.md](docs/benchmarks/drone-test.md).

## Docker (Linux + NVIDIA GPU)

A Linux image on NVIDIA's TensorRT container builds everything (TensorRT and CUDA frame prep included) and runs the video-file benchmarks with one command: timing, MOT17 against ByteTrack, and Anti-UAV dropout. It needs Docker with GPU access (`--gpus all`) and the datasets in `data/`:

```
docker build -f docker/Dockerfile -t tether .
docker run --rm --gpus all -v ./data:/tether/data:ro -v ./models/docker:/tether/models -v ./results/docker:/tether/results tether
```

Datasets, models, other GPUs and running the tests in the container: [docs/DOCKER.md](docs/DOCKER.md). No webcam or window inside Docker.

## Design

The full spec is in [docs/SPEC.md](docs/SPEC.md).

## License

AGPL-3.0, to match the Ultralytics model license.
