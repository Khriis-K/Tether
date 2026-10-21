# The TensorRT runtime: ONNX Runtime against TensorRT on the same videos (#16)

**Question:** with the same YOLO26n model run by TensorRT on the laptop GPU instead of ONNX Runtime on the CPU, how much faster is the model run, and are the detections still the same?

**Answer:** the model run (the pipeline's `detect` step) is **5.16–5.63× faster at 640** (32.77–33.33 → 5.84–6.46 ms) and **10.01–10.45× faster at 1280×736** (72.68–76.26 → 7.07–7.36 ms). That clears the spec's 1.5× by a wide margin. Whole-pipeline FPS (sequential, headless) goes up **2.15–3.25× at 640** and **3.24–4.66× at 1280×736**, less than detect alone because capture and prep don't change. The detections **agree within the tolerance on all four video × input-size pairs**: every detection scoring at least 0.25 in one runtime had a partner in the other, with scores at most 0.0003 apart and box corners at most 0.001 px apart. That's for engines built with `--noTF32`. With `trtexec`'s default, which allows TF32, MOT17-04 at 1280×736 **failed** the check (98.91% partnered against the 99% needed), because 1.09% of the detections checked had a partner box but a score more than 0.02 away. Turning TF32 off cost at most 0.335 ms a frame, so the documented engine build uses `--noTF32`.

Caveat on what's compared: ONNX Runtime here is its **CPU** provider (the only one in the pipeline), so this is the GPU against the CPU, not TensorRT against ONNX Runtime's CUDA provider. Most of the gain is the GPU; how much is TensorRT's compilation wasn't measured.

## Setup

- Code: the TensorRT detector as committed for #16 (`cpp/src/tether/tensorrt_detector.cpp`), behind the same `Detector` interface as `OnnxDetector`, picked with `tether-pipeline --runtime onnxruntime|tensorrt`. Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB RAM, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power.
- Runtimes: ONNX Runtime 1.31.0, CPU provider, default threads (as in [#11](cpp-pipeline-speed.md) and [#14](parallel-pipeline.md)). TensorRT 11.4.0.106 (the CUDA 13.4 build), CUDA runtime 13.4 linked statically.
- Engines: built from the same ONNX files with `trtexec --onnx=models/<model>.onnx --saveEngine=models/<model>.engine --noTF32 --skipInference`, one per input size (`yolo26n.onnx` at 640×640 and `yolo26n_1280x736.onnx`). Each took 67–90 s to build. TensorRT 11 builds strongly typed networks, so both engines are float32 like the ONNX model ([Strongly typed network](../CONCEPTS.md#strongly-typed-network-precision-from-the-model)). The TF32 rows below used engines built the same way without `--noTF32`.
- What `detect` covers for TensorRT: copying the prepared frame from ordinary (pageable) host memory to the GPU, running the engine, copying the 300×6 output back, waiting for the stream, and decoding. For ONNX Runtime it's the session run and the decoding. Frame prep is the CPU one in both.
- Command: `python -m tether.runtime_compare VIDEO --model models/<model>.onnx --rounds 2`. It first runs `tether-pipeline --runtime R --dump` once per runtime and checks agreement on the two detection files, then runs `tether-pipeline --runtime R --timing` (sequential, headless) twice per runtime, alternating which goes first each round, as `tether-mode-compare` does. Every frame is timed, including the first. Summaries are `tether.timing.summarize`.
- Videos: the same two as #11 and #14. The Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames).
- Results files: `results/runtime-compare/{antiuav-yolo26n, antiuav-yolo26n_1280x736, notf32-mot17-04-yolo26n, notf32-mot17-04-yolo26n_1280x736}.json` (engines built with `--noTF32`) and `results/runtime-compare/tf32/*.json` (TF32 allowed), each with its detection dumps and timing logs next to it. All eight are copied into [`tensorrt-runtime.json`](tensorrt-runtime.json) so they're kept with the repo (`results/` is ignored).

## Agreement

The tolerance was set in `tether/runtime_compare.py` before the first comparison on the real model: every detection scoring at least 0.25 (the tracker's own threshold) in either runtime must have a partner in the other, of the same class, with IoU ≥ 0.95 and scores within 0.02. The runtimes agree when at least 99% of those detections have one. The partner may score below 0.25, so a detection sitting on the threshold isn't counted missing because it landed just under it in the other runtime ([Detection agreement](../CONCEPTS.md#detection-agreement-between-runtimes)).

| Video | Input | Engine | Checked | With a partner | Worst partner score diff | Worst partner box diff | Result |
|---|---|---|---|---|---|---|---|
| Anti-UAV | 640 | `--noTF32` | 56 | 56 (100%) | 0.0000 | 0.001 px | pass |
| Anti-UAV | 1280×736 | `--noTF32` | 4 | 4 (100%) | 0.0000 | 0.000 px | pass |
| MOT17-04 | 640 | `--noTF32` | 30,214 | 30,214 (100%) | 0.0001 | 0.001 px | pass |
| MOT17-04 | 1280×736 | `--noTF32` | 79,268 | 79,268 (100%) | 0.0003 | 0.001 px | pass |
| Anti-UAV | 640 | TF32 | 55 | 55 (100%) | 0.0186 | 2.814 px | pass |
| Anti-UAV | 1280×736 | TF32 | 4 | 4 (100%) | 0.0037 | 0.239 px | pass |
| MOT17-04 | 640 | TF32 | 30,235 | 30,031 (99.33%) | 0.0198 | 3.352 px | pass |
| MOT17-04 | 1280×736 | TF32 | 79,277 | 78,413 (98.91%) | 0.0200 | 1.421 px | **fail** |

"Checked" counts each detection from both sides, so it's about twice the detections per runtime. The Anti-UAV clip has very few detections at 0.25 or more (a small drone, and a COCO model with no drone class), so its rows say little; MOT17-04 carries this check.

**What failed with TF32.** All 864 detections without a partner on MOT17-04 at 1280×736 (and all 204 at 640) failed on score alone: each had a same-class box in the other runtime overlapping by IoU 0.95 or more. Across all detections checked at 640, the median score difference was 0.0004 and the 99th percentile 0.013, but the largest was 0.315, on one person at x≈1570 px around frames 707–775, where YOLO26 put two near-identical boxes and the two runtimes split the score between them differently (frame 707: 0.536 on ONNX Runtime, 0.221 on TensorRT, IoU 0.9996). At 1280×736, 355 of the 79,277 differed by more than 0.05, and the largest by 0.263. (These breakdowns were computed from the dumps in `results/runtime-compare/tf32/` after the runs; they aren't in the results files.)

**Without TF32 the outputs are near-identical.** Not byte-identical: not one frame's line in the dump files is the same text, because the last digits of boxes and scores differ, but both runtimes kept exactly the same number of detections at 0.1 and above on MOT17-04 at 640 (26,212 each). ONNX Runtime itself was deterministic: its dumps were byte-identical between the TF32 and `--noTF32` sessions for every video and size.

The C++ unit tests check the same thing on the stand-in model, where both runtimes must give exactly the same detection (`TensorRtDetector.GivesTheSameDetectionsAsOnnxRuntime`, and `tether-pipeline --runtime tensorrt --dump` against ONNX Runtime's dump).

## Timing

Mean milliseconds per frame (P99 in brackets for frame time and detect). Sequential pipeline, headless, engines built with `--noTF32`. "ORT" is ONNX Runtime (CPU), "TRT" TensorRT (GPU).

| Video, input, round | | FPS | Frame time | capture | prep | detect | track | End-to-end delay |
|---|---|---|---|---|---|---|---|---|
| Anti-UAV 640, 1 | ORT | 25.2 | 39.73 (45.80) | 2.72 | 3.49 | 33.02 (38.45) | 0.01 | 36.53 |
| | TRT | 71.5 | 13.98 (20.89) | 3.51 | 4.09 | 5.86 (11.19) | 0.01 | 9.96 |
| Anti-UAV 640, 2 | TRT | 79.9 | 12.52 (18.51) | 2.75 | 3.55 | 5.84 (11.32) | 0.01 | 9.40 |
| | ORT | 24.5 | 40.74 (47.19) | 3.37 | 3.94 | 32.77 (38.46) | 0.01 | 36.72 |
| Anti-UAV 1280×736, 1 | ORT | 11.9 | 84.21 (89.56) | 2.67 | 7.61 | 72.93 (78.17) | 0.01 | 80.55 |
| | TRT | 54.8 | 18.24 (23.57) | 2.72 | 7.53 | 7.21 (11.87) | 0.01 | 14.75 |
| Anti-UAV 1280×736, 2 | TRT | 55.6 | 17.98 (23.31) | 2.68 | 7.46 | 7.07 (12.03) | 0.01 | 14.54 |
| | ORT | 11.9 | 83.86 (89.73) | 2.67 | 7.58 | 72.68 (77.65) | 0.01 | 80.27 |
| MOT17-04 640, 1 | ORT | 20.0 | 50.06 (57.86) | 12.38 | 3.49 | 33.33 (39.76) | 0.37 | 37.19 |
| | TRT | 43.0 | 23.28 (30.60) | 12.39 | 3.64 | 6.46 (12.16) | 0.35 | 10.45 |
| MOT17-04 640, 2 | TRT | 43.5 | 22.96 (29.87) | 12.27 | 3.63 | 6.32 (11.51) | 0.33 | 10.28 |
| | ORT | 20.0 | 50.03 (55.21) | 12.35 | 3.52 | 33.30 (38.14) | 0.37 | 37.18 |
| MOT17-04 1280×736, 1 | ORT | 9.8 | 102.21 (179.18) | 13.88 | 9.61 | 76.26 (132.52) | 0.91 | 86.78 |
| | TRT | 34.1 | 29.35 (46.36) | 11.80 | 8.23 | 7.30 (11.99) | 0.78 | 16.31 |
| MOT17-04 1280×736, 2 | TRT | 34.0 | 29.42 (38.19) | 12.22 | 8.06 | 7.36 (13.32) | 0.81 | 16.23 |
| | ORT | 10.5 | 95.46 (101.11) | 12.40 | 7.66 | 73.66 (79.26) | 0.89 | 82.21 |

Speedups (ORT time over TRT time; FPS is TRT's over ORT's), rounds 1 and 2:

| Video, input | detect | FPS |
|---|---|---|
| Anti-UAV 640 | 5.63×, 5.61× | 2.84×, 3.25× |
| Anti-UAV 1280×736 | 10.11×, 10.28× | 4.62×, 4.66× |
| MOT17-04 640 | 5.16×, 5.27× | 2.15×, 2.18× |
| MOT17-04 1280×736 | 10.45×, 10.01× | 3.48×, 3.24× |

### TF32 allowed (`trtexec` default)

Same commands, engines built without `--noTF32`, run in an earlier session the same evening:

| Video, input | ORT detect, rounds 1, 2 | TRT detect, rounds 1, 2 | detect speedup | FPS speedup |
|---|---|---|---|---|
| Anti-UAV 640 | 35.40, 35.39 | 6.05, 5.76 | 5.85×, 6.14× | 3.20×, 3.54× |
| Anti-UAV 1280×736 | 76.78, 76.58 | 6.80, 6.81 | 11.30×, 11.25× | 4.77×, 4.70× |
| MOT17-04 640 | 34.50, 34.76 | 6.25, 6.29 | 5.52×, 5.52× | 2.30×, 2.30× |
| MOT17-04 1280×736 | 76.60, 76.38 | 7.22, 7.15 | 10.60×, 10.68× | 3.19×, 3.19× |

Comparing TensorRT's mean detect time between the two builds (average of the two rounds each): `--noTF32` was 0.06 ms faster on Anti-UAV at 640 (5.85 against 5.91 ms), and 0.12, 0.335 and 0.145 ms slower on MOT17-04 at 640, Anti-UAV at 1280×736 and MOT17-04 at 1280×736 (up to 4.9%). The two builds ran in different sessions, and ONNX Runtime's own mean detect time moved by 1.3–3.9 ms between them on the same model and video, so these differences are within the drift between sessions. The TF32 build's higher detect speedups come mostly from the slower ONNX Runtime runs in that session, not a faster TensorRT.

## Reading the numbers

- **The detector stopped being the bottleneck.** On MOT17-04 at 640, TensorRT's frame (22.96–23.28 ms) is now capture 12.27–12.39, prep 3.63–3.64, detect 6.32–6.46 and track 0.33–0.35 ms. Decoding the JPEGs (`capture`) now takes nearly twice as long as the model run, and on the Anti-UAV clip prep (3.55–4.09 ms) is not far behind it. So the whole-pipeline gain is well under the detect gain: 2.15–2.18× FPS for a 5.16–5.27× faster detect on MOT17-04. That's where #17 (CUDA frame prep) and the parallel pipeline come in: in parallel mode the frame time was the detect time ([#14](parallel-pipeline.md)), and with detect at 6 ms the slowest stage would be capture instead. The parallel pipeline with TensorRT wasn't measured here.
- **Detect barely grows with input size on the GPU.** 1280×736 has 2.3× the pixels of 640×640, and ONNX Runtime's detect went up 2.21–2.25× (averages of the two rounds), but TensorRT's only 1.15–1.22× (5.84–6.46 → 7.07–7.36 ms). That suggests most of TensorRT's 6 ms at 640 is fixed cost (copies, launches, waiting) rather than arithmetic. The copy alone moves 4.9 MB of float32 input at 640 and 11.3 MB at 1280×736 from pageable memory. Splitting the 6 ms into copy, compute and wait needs a profiler (#15, Nsight Systems).
- **TensorRT's P99 is 1.6–1.9× its mean.** Detect P99 is 11.19–13.32 ms against means of 5.84–7.36 ms, a wider spread than ONNX Runtime's (P99 1.07–1.19× its mean, except MOT17-04 1280×736 round 1, whose 132.52 ms P99 also shows in its 179.18 ms frame-time P99). Every frame is timed, including the first, which includes the engine's first launch, but one frame can't move a P99 over 1000 frames. Why the GPU's slow frames are slow wasn't looked into here; #15 can see it.
- **End-to-end delay drops with the model run.** Sequentially, delay is the time from capture to the end of draw, so it falls from 36.53–37.19 ms to 9.40–10.45 ms at 640, and from 80.27–86.78 ms to 14.54–16.31 ms at 1280×736.
- **The spec's 120+ FPS on video files is still out of reach sequentially.** The best here is 79.9 FPS (Anti-UAV, 640). On MOT17-04, capture alone (12.27–12.39 ms) caps a sequential pipeline at about 81 FPS, before any other step.
