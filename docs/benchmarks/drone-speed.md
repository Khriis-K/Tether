# The drone model in the shipped configuration: speed and runtime agreement (#46)

**Question:** the drone model (`models/yolo26n_drone_1280x736.onnx`, #43) has the shipped detector's shape, YOLO26n at 1280×736, so its speed should carry over. Does it actually meet the spec's targets, 120+ FPS and P99 frame time under 16 ms, in the shipped configuration (`--runtime tensorrt --prep cuda --pipeline parallel`)? And does its TensorRT engine give ONNX Runtime's detections?

**Answer: yes to both.** On the val clip, the drone model runs at **195.59–198.63 FPS with P99 6.46–6.49 ms** in the shipped configuration, meeting both targets in both rounds. The COCO model on the same clip in the same session runs 195.95–197.18 FPS (P99 6.12–6.26 ms). Both are limited by **capture (video decoding), busy 99.7%**, so the detector doesn't set the FPS. The drone model's detect is 2.91–2.92 ms against COCO's 2.76–2.77, about 0.15 ms slower, which doesn't show in FPS while decoding is the limit. Its TensorRT engine **agrees with ONNX Runtime on all 2,002 checked detections** (worst score difference 0.000058).

Raw results: [`drone-speed.json`](drone-speed.json) holds the `tether-model-compare` run (every run's timing summary and busy shares) and three `tether-runtime-compare` runs (agreement and timing rounds).

## The clip changed

Every earlier speed number used the Anti-UAV clip `test/20190925_111757_1_1` ([cuda-prep](cuda-prep.md), [model-size](model-size.md)). That clip is in Anti-UAV **test**, which the drone model must not see before the one final run (#48, [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md)), not even for timing. So this run uses **`train/20190925_131530_1_1`**, the first clip of the session-disjoint val in `splits/antiuav.json`, picked by that rule rather than by its speed. Like the old clip, it's 1920×1080 at 20 fps, 1000 frames, with the same codec (MPEG-4 Part 2, FourCC `FMP4`), at a higher bitrate: 2.48 Mb/s against 1.87.

So the numbers here aren't the same measurement as `cuda-prep.md`'s 195.45–195.86 FPS. The COCO model rows were timed on the new clip in the same session for a like-for-like comparison. They came out within the earlier clip's range. (`cuda-prep.md` and `model-size.md` call the old clip H.264; OpenCV reports `FMP4`, MPEG-4 Part 2, for both clips.)

## Setup

- Code as committed for #46. TensorRT 11.4.0.106 with the CUDA graph on (the default), CUDA 13.4. Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power. Headless. No other GPU work ran during the timing runs.
- **Engines:** `trtexec --onnx=M.onnx --saveEngine=M.engine --noTF32 --skipInference`. The drone engine was rebuilt for this issue from `yolo26n_drone_1280x736.onnx`, which is byte-identical to #43's `yolo26n_drone_1280x736_e19.onnx`. The COCO engine is #16/#18's. Engines stay out of git. The rebuilt drone engine gives #43's accuracy: its val dump agrees with the old engine's on all 76,606 checked detections and scores the same AP50 and found counts ([drone-tracking](drone-tracking.md#the-c-pipeline-gives-the-same-result-46), `drone-model-rebuilt-engine.json`).
- **Speed:** `tether-model-compare VIDEO --models models/yolo26n_drone_1280x736.onnx models/yolo26n_1280x736.onnx --prep cuda --pipeline parallel --rounds 2`. Each model runs twice, the order rotating (drone first in round 1, COCO first in round 2), and every frame is timed, including the first. A target counts as met only if both rounds met it. `--prep` is new in this issue: until now the tool only ran CPU prep.
- **Agreement:** `tether-runtime-compare VIDEO --model models/yolo26n_drone_1280x736.onnx` (ONNX Runtime against TensorRT, two sequential timing rounds), with #16's tolerance: every detection scoring 0.25 or more on one side needs a partner on the other, of the same class, at IoU 0.95 or more and scores at most 0.02 apart, for at least 99% of them.
- **CPU against CUDA prep:** `tether-runtime-compare VIDEO --compare frame-prep --pipeline parallel --rounds 2`, once per model, as in [cuda-prep](cuda-prep.md).

## Shipped configuration (TensorRT, CUDA prep, parallel)

Ranges are min–max over the two rounds.

| Model | FPS | Frame time ms | P99 ms | Detect ms | End-to-end ms | Slowest stage (busy) | 120+ FPS | P99 < 16 ms |
|---|---|---|---|---|---|---|---|---|
| **yolo26n_drone_1280x736** | **195.59–198.63** | 5.03–5.11 | **6.46–6.49** | 2.91–2.92 | 3.92–3.94 | **capture (99.7%)** | **yes** | **yes** |
| yolo26n_1280x736 (COCO) | 195.95–197.18 | 5.07–5.10 | 6.12–6.26 | 2.76–2.77 | 3.77 | capture (99.7%) | yes | yes |

Busy shares, drone model: capture 99.7%, detect 56.5–57.2%, prep 17.3–17.4%, track 0.6%. COCO: capture 99.7%, detect 53.8–53.9%, prep 17.1–17.5%, track 0.2%.

## Runtime agreement (ONNX Runtime against TensorRT)

| Model | Checked (score ≥ 0.25) | With a partner | Worst score difference | Worst IoU | Dumps identical |
|---|---|---|---|---|---|
| yolo26n_drone_1280x736 | 2,002 | 2,002 (100%) | 0.000058 | 0.999998 | no |

**PASS.** The dumps differ only in the last digits, as with the COCO model (#16): kernels add numbers in a different order ([Detection agreement](../CONCEPTS.md#detection-agreement-between-runtimes)). For context, the same run's sequential timing, with CPU prep: TensorRT detect 4.25–4.41 ms against ONNX Runtime's 67.83–68.11 (15.39–16.04×), FPS 67.60–69.19 against 12.64–12.71.

## CPU against CUDA prep (parallel)

| Model | FPS, CPU → CUDA prep | P99 ms, CPU → CUDA | Prep ms, CPU → CUDA | Detections |
|---|---|---|---|---|
| drone | 126.31–126.42 → 195.35–195.66 | 9.30–9.76 → 6.41–6.66 | 7.92 → 0.88–0.90 | byte-identical (2,002 checked) |
| COCO | 126.95–127.88 → 190.68–201.66 | 9.15–9.20 → 6.03–8.23 | 7.83–7.88 → 0.87–0.88 | byte-identical (64 checked) |

These reproduce `cuda-prep.md`'s pattern on the new clip: CUDA prep moves the bottleneck from prep to decoding, and gives the model the same input. The COCO side's wide CUDA-prep range (190.68–201.66 FPS, P99 up to 8.23 ms) is one round each way. Capture was the slowest stage in both, so the decoder's pace moved between rounds. This wasn't investigated.

## What it means

- **The drone model meets the speed targets with room to spare**: its slowest round is 75.59 FPS over 120, and its worst P99, 6.49 ms, is 9.51 ms inside the 16 ms limit. Like the COCO model, it's decode-bound, so a faster detector wouldn't raise FPS on this clip, and a slower one would cost nothing until detect passed capture's 5.06–5.14 ms.
- **Detect is about 0.15 ms slower than the COCO model's** (2.91–2.92 against 2.76–2.77 ms, and P99 3.28–3.31 against 2.93–2.98). The two models have the same layers and input. A possible cause is the engine build: TensorRT times candidate kernels while it builds and keeps the fastest, so two builds of the same network can choose differently. That wasn't tested.
- **The track step is 0.03 ms for the drone model against 0.01 for COCO.** On this clip the drone model finds the drone on most frames, and the COCO model rarely does (64 detections at 0.25 or more over 1000 frames, against 2,002).

## Caveats

- One laptop, two rounds per configuration, one session, on a desktop that wasn't idle.
- One clip, chosen as the first val clip, not as typical. A val clip is from the sessions the drone model's checkpoint and threshold were chosen on, but timing doesn't depend on that.
- Agreement on this clip rests on 2,002 detections of one drone. The COCO model's agreement on MOT17-04 (#16) covers crowded scenes; the drone model wasn't checked there.
- Headless only. `--show` with the drone model was checked by its unit tests (the panel names class 0 "drone"), not timed.
- The tracker inside these runs uses `tether-pipeline`'s default cutoff, 0.25, not the drone model's 0.352: `tether-pipeline` has no cutoff flag. The track step is 0.6% busy, so the cutoff can't move FPS here.

## Reproduce

From the main checkout, with a TensorRT build of `tether-pipeline` in `cpp/build`:

```
trtexec --onnx=models/yolo26n_drone_1280x736.onnx --saveEngine=models/yolo26n_drone_1280x736.engine --noTF32 --skipInference
tether-model-compare data/Anti-UAV300/train/20190925_131530_1_1/visible.mp4 \
    --models models/yolo26n_drone_1280x736.onnx models/yolo26n_1280x736.onnx --prep cuda --pipeline parallel --rounds 2
tether-runtime-compare data/Anti-UAV300/train/20190925_131530_1_1/visible.mp4 --model models/yolo26n_drone_1280x736.onnx
tether-runtime-compare data/Anti-UAV300/train/20190925_131530_1_1/visible.mp4 --compare frame-prep --pipeline parallel \
    --model models/yolo26n_drone_1280x736.onnx
```

The COCO frame-prep row is the last command with `--model models/yolo26n_1280x736.onnx`.
