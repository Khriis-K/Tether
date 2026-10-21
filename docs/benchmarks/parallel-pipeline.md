# The parallel pipeline: sequential vs parallel on the same videos (#14)

**Question:** with capture, prep, detect and track each on their own thread, joined by bounded queues, how much faster is the C++ pipeline than the sequential one on the same video? Does it still give the same tracks? What does it do to end-to-end delay? And does the webcam keep up now?

**Answer:** with the one-frame queues it ships with, parallel is **1.23–1.24× faster on the Anti-UAV clip** (24.2–24.4 → 30.0–30.1 FPS) and **1.57× on MOT17-04** (19.3 → 30.3 FPS), YOLO26n at 640. Earlier rounds with two-frame queues gave 1.56–1.62× on MOT17-04, and on Anti-UAV 1.15–1.23× plus one 0.65× outlier (see below). Both are at the ceiling a parallel pipeline can reach with this detector. Its frame time settles at the detect time (about 33 ms), so the speedup can't go much past sequential frame time over detect time: 1.26× on Anti-UAV and 1.54–1.57× on MOT17-04. The tracks are **the same**: recordings with the tracks drawn came out byte-identical between the two modes on MOT17-09 (525 frames) and on the Anti-UAV clip (1000 frames). The price is **end-to-end delay**. On a video file the stages before the detector keep a frame waiting in each queue, so delay goes from about 37 ms to **147–159 ms** with one-frame queues (213–242 ms with two-frame queues, which is why the queues hold one). On the laptop webcam, parallel **dropped 0 frames in 300** against 129 for sequential, at 29.9 FPS against a 30 FPS camera. That's with only 0.2 ms to spare, though: detect took 33.16 ms against a 33.33 ms camera period.

## Setup

- Code: the parallel pipeline as committed for #14 (`run_parallel` in `cpp/src/tether/pipeline.cpp`). Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB RAM, on mains power. ONNX Runtime 1.31.0 (CPU provider, default threads) and OpenCV 5.0.0, as in [#11](cpp-pipeline-speed.md).
- The two-frame-queue runs were built before one last change that only affects error handling (a stopped stage no longer closes its input queue as well as its output; the end of the run closes every queue anyway). The one-frame-queue runs, the recordings and the webcam runs used the final code.
- Command: `python -m tether.mode_compare VIDEO --model models/yolo26n.onnx --rounds N`. It runs `tether-pipeline --pipeline sequential` and `--pipeline parallel` headless on the same file, each in its own process, and alternates which mode goes first each round, so a laptop speeding up or slowing down during the runs lands on both modes. Every frame is timed, including warm-up. Summaries are `tether.timing.summarize`.
- Videos: the Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames), the same two as [#11](cpp-pipeline-speed.md).
- The machine wasn't idle. The desktop apps (a vendor utility, the browser, this app) used about two cores' worth of CPU before the runs, and that can't be turned off from here.
- Results files: `results/mode-compare/{antiuav-640, mot17-04-640, cap1-antiuav-640, cap1-mot17-04-640}.json`, with every run's timing log next to them. Their summaries are copied into [`parallel-pipeline.json`](parallel-pipeline.json) so they're kept with the repo (`results/` is ignored).

## Results

Mean milliseconds per frame (P99 in brackets for frame time and delay). "Seq" is sequential, "par" parallel.

### One-frame queues (as shipped, `PARALLEL_QUEUE_CAPACITY = 1`)

| Video, round | | FPS | Frame time | End-to-end delay | capture | prep | detect | track |
|---|---|---|---|---|---|---|---|---|
| Anti-UAV, 1 | seq | 24.2 | 41.33 (45.53) | 36.98 (40.70) | 3.66 | 4.12 | 32.85 | 0.01 |
| | par | 30.0 | 33.39 (37.80) | 158.94 (166.94) | 7.81 | 5.14 | 33.37 | 0.01 |
| Anti-UAV, 2 | seq | 24.4 | 40.93 (45.59) | 36.78 (40.74) | 3.46 | 3.97 | 32.80 | 0.01 |
| | par | 30.1 | 33.28 (37.65) | 158.57 (169.76) | 7.64 | 5.10 | 33.26 | 0.01 |
| MOT17-04, 1 | seq | 19.3 | 51.90 (66.34) | 38.02 (52.52) | 13.19 | 4.02 | 33.63 | 0.37 |
| | par | 30.3 | 32.95 (36.20) | 146.96 (153.85) | 18.01 | 4.70 | 32.93 | 0.47 |
| MOT17-04, 2 | seq | 19.3 | 51.93 (57.32) | 37.66 (42.88) | 13.47 | 4.16 | 33.13 | 0.36 |
| | par | 30.3 | 33.05 (36.32) | 147.21 (154.39) | 18.23 | 4.66 | 33.04 | 0.44 |

Speedup (parallel FPS over sequential): Anti-UAV **1.238×, 1.230×**; MOT17-04 **1.575×, 1.571×**. Delay, parallel over sequential: 4.30×, 4.31×; 3.87×, 3.91×.

### Two-frame queues (the first version)

| Video, round | | FPS | Frame time | End-to-end delay | capture | prep | detect | track |
|---|---|---|---|---|---|---|---|---|
| Anti-UAV, 1 | seq | 22.9 | 43.70 (94.67) | 39.08 (84.49) | 3.95 | 4.23 | 34.83 | 0.01 |
| | par | **14.8** | 67.50 (329.34) | 463.20 (1473.66) | 8.35 | 6.03 | **67.36** | 0.02 |
| Anti-UAV, 2 | seq | 24.2 | 41.34 (49.55) | 37.04 (45.07) | 3.52 | 4.02 | 33.01 | 0.01 |
| | par | 27.9 | 35.84 (58.51) | 242.14 (385.61) | 8.36 | 5.25 | 35.82 | 0.01 |
| Anti-UAV, 3 | seq | 24.2 | 41.32 (45.60) | 36.99 (40.84) | 3.58 | 4.04 | 32.94 | 0.01 |
| | par | 29.8 | 33.58 (38.98) | 227.42 (240.42) | 7.17 | 4.87 | 33.57 | 0.01 |
| MOT17-04, 1 | seq | 18.6 | 53.76 (63.87) | 37.93 (47.10) | 15.06 | 4.19 | 33.36 | 0.38 |
| | par | 30.1 | 33.20 (36.63) | 213.42 (222.90) | 18.80 | 4.73 | 33.19 | 0.47 |
| MOT17-04, 2 | seq | 19.4 | 51.67 (57.18) | 37.54 (42.60) | 13.37 | 4.12 | 33.05 | 0.37 |
| | par | 30.2 | 33.08 (37.16) | 213.63 (221.29) | 17.72 | 4.46 | 33.06 | 0.47 |
| MOT17-04, 3 | seq | 19.2 | 51.97 (56.45) | 37.73 (42.32) | 13.48 | 4.25 | 33.11 | 0.37 |
| | par | 30.1 | 33.20 (36.96) | 214.09 (222.15) | 18.17 | 4.67 | 33.22 | 0.46 |

Speedup: Anti-UAV 0.647×, 1.153×, 1.231×; MOT17-04 1.619×, 1.562×, 1.565×.

### Same tracks

`tether-pipeline VIDEO --record OUT.avi`, once per mode, then every frame of the two recordings compared. The recording has the HUD drawn on it, so it shows every track's box, ID label, colour, trail and status.

| Video | Frames | Frames that differ | Files |
|---|---|---|---|
| MOT17-09 (`img1/%06d.jpg`) | 525 | 0 | byte-identical |
| Anti-UAV clip | 1000 | 0 | byte-identical |

The unit tests check the same thing on scripted scenes with random waits in every stage, three interleavings per run (`RunParallel.GivesTheSameTracksInTheSameFrameOrderAsSequential`), and `tether-pipeline --pipeline parallel --record` against sequential on the stand-in model.

### Webcam

The laptop's built-in camera (30 FPS), YOLO26n at 640, `--max-frames 300`, one run per mode, sequential first:

| | FPS | Frame time | End-to-end delay | capture | detect | Dropped by the camera |
|---|---|---|---|---|---|---|
| Sequential | 20.9 | 47.90 (106.83) | 37.40 (43.23) | 11.33 | 33.53 | 129 |
| Parallel | 29.9 | 33.47 (41.72) | 82.97 (138.48) | 34.75 | 33.16 | **0** |

Checking the drop counts independently from the capture times: the sequential run's 300 frames span 14,335 ms, about 430 camera periods, so about 130 went missing. The parallel run's span is 9,970 ms, about 299 periods, so none did. Webcam delay in parallel is lower than on a file (83 against 147–159 ms), because capture can't run ahead of the camera: it waits for each frame (that wait is the 34.75 ms `capture`), so the queues in front of the detector rarely fill.

## Reading the numbers

- **The detector sets the pace, now on its own.** In parallel, the frame time (32.95–33.58 ms in the good rounds) is the detect time (32.93–33.57 ms). Every other step now runs while the detector works on the frame ahead. So the most parallel can win is sequential frame time over detect time: 41.33/32.85 = 1.26× on Anti-UAV and 51.90/33.63 = 1.54× to 51.93/33.13 = 1.57× on MOT17-04. The measured 1.23× and 1.57× are at that ceiling. MOT17-04's round 1 lands just above its 1.54×, because the parallel run's `detect` happened to be faster (32.93 against 33.63 ms). MOT17-04 gains more because it has more to hide: decoding its JPEGs (`capture`, 13–15 ms) is far slower than decoding the Anti-UAV MPEG-4 (3.5 ms). The spec's 120+ FPS on video files needs a detector under 8.3 ms, so TensorRT, not more threads.
- **Delay is frames waiting times the detect time ([Little's law](../CONCEPTS.md#littles-law)).** On a file, capture and prep are faster than detect, so they run ahead until the queues before the detector are full, and every frame then waits its turn. With one-frame queues that's about 4.5–4.8 detect times (147–159 ms over 33 ms): capture holds one frame waiting to push, one sits in the queue to prep, prep holds one, one sits in the queue to detect, and one is in detect. Two-frame queues add two more waiting frames, and delay went up by about two detect times (to 213–242 ms) while FPS stayed the same. That's the measurement behind `PARALLEL_QUEUE_CAPACITY = 1`.
- **Parallel steps run a little slower.** `capture` roughly doubles on Anti-UAV (3.5 → 7.7 ms) and goes up a third on MOT17-04 (13.3 → 18.1 ms), and `prep` rises by 0.5–1 ms. Decoding now runs at the same time as ONNX Runtime, which keeps its own pool of threads busy on the cores. It costs nothing in FPS here, because those steps aren't the bottleneck, but they would be if the detector got faster.
- **The outlier round.** In the first two-frame round on Anti-UAV, parallel ran at 14.8 FPS because `detect` itself took 67.36 ms on average (P99 frame time 329 ms), twice its usual time. The sequential run just before it was also slower than usual (detect 34.83, P99 frame time 94.67 ms). The cause wasn't found. Something else on the machine is the likely suspect, since `detect` is the same call in both modes, and the five later parallel runs on that clip were all at 27.9–30.1 FPS. It's left in the table rather than dropped.
- **The webcam keeps up, but only just.** At 30 FPS the camera gives a new frame every 33.33 ms, and parallel needs one every detect time, 33.16 ms here. A detector 0.5% slower (a warmer laptop, a busier desktop) would drop frames again. Sequential needed 47.90 ms a frame, so it fell behind by a third and the camera dropped 129 frames (69 in #11's run, on a different day).

## Caveats

- One laptop. Two or three rounds per video and queue size, and one webcam run per mode. Apart from the outlier, parallel ran at 27.9–30.1 FPS on Anti-UAV and 30.1–30.3 on MOT17-04, but one parallel run in ten was a 2× outlier.
- The desktop wasn't idle (about two cores' worth of other programs), the same for both modes.
- Headless only: `draw` was 0 ms in every run. With `--show`, `draw` is 10–14 ms ([#13](stats-panel.md)), which runs alongside detect in parallel but adds to every frame in sequential, so the speedup with a window should be larger. That hasn't been measured.
- The thread count wasn't tuned. ONNX Runtime uses its default intra-op threads in both modes, and the pipeline adds four threads in parallel mode. How ONNX Runtime's thread count trades against the decode slowdown isn't known.
- Only YOLO26n at 640. At 1280×736 the detector is about 74 ms ([#11](cpp-pipeline-speed.md)), so detect would dominate even more and the speedup would be smaller. Not measured.
- Pausing the window (`--show`, Space) in parallel mode holds the frames already read: up to eight, one in each of the four queues and one held by each of the four stage threads. After resuming they're shown first, and on a webcam they arrive with a delay as long as the pause. Sequential holds only the paused frame.
- On a file the pipeline reads ahead and drops nothing, so the extra delay buys no frames. A webcam pipeline that cared about delay would drop the oldest frame instead of waiting. That isn't built.
