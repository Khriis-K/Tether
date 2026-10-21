# A CUDA graph for TensorRT's enqueue: the engine replayed instead of relaunched (#40)

**Question:** #15 found TensorRT's detect is launch-bound: at 640 the CPU spent 5.81–6.10 ms (profiled) launching the engine's 217 kernels one by one, while the GPU computed for 1.47–1.53 ms and idled 4.29–4.58 ms. If `TensorRtDetector` captures the engine into a [CUDA graph](../CONCEPTS.md#cuda-graph-capture-once-replay-each-frame) once and replays it each frame, how much faster is detect, and does the pipeline get faster?

**Answer:** detect is **2.10–2.24× faster at 640** (5.10–6.37 → 2.32–2.97 ms) and **1.49–1.64× at 1280×736** (6.31–7.31 → 4.02–4.60 ms). That held on both videos, in sequential and parallel mode, with **byte-identical detections** with and without the graph. Profiled, launching the engine now takes 0.407–0.438 ms at 640 instead of 5.834 ms (a no-graph capture in the same session), and the GPU idles 0.334–0.378 ms a detect instead of 4.376 ms. Detect P99 at 640 drops from 7.38–11.87 ms to 2.83–7.40 ms.

What the pipeline gets depends on whether detect was the slowest step:
- **Sequential** (every step adds up): FPS **1.35–1.37×** on the Anti-UAV clip at 640 (81.08–84.14 → 111.81–113.97) and 1.15–1.16× on MOT17-04, 1.08–1.19× at 1280×736.
- **Parallel** (FPS set by the slowest stage): **1.10× on the Anti-UAV clip at 640** (172.57–173.43 → 191.30–191.99 FPS, P99 frame time 9.57–9.67 → 6.59–7.11 ms), and 1.02–1.04× elsewhere. The slowest stage (capture, or prep on the Anti-UAV clip at 1280×736) was the same before and after, busy 99.2–99.9%. Detect's busy share halved (87.4–87.9% → 44.0–44.4% on Anti-UAV at 640).

The graph is on by default; `tether-pipeline --no-cuda-graph` turns it off. Copying the input through pinned memory was not built: the copies are now a large part of what's left of detect (0.47–0.54 ms of CPU at 640), but detect isn't the slowest stage in parallel mode, so a faster copy couldn't raise its FPS.

## Setup

- Code: `TensorRtDetector` as committed for #40 (`cpp/src/tether/tensorrt_detector.cpp`). At construction it runs the engine once uncaptured (TensorRT's one-off setup), then records one `enqueueV3` on its own stream with `cudaStreamBeginCapture`/`cudaStreamEndCapture` (thread-local capture mode) and instantiates the graph. Each `detect()` then copies the frame in, calls `cudaGraphLaunch`, and copies the rows back, as before. The copies stay outside the graph. If capture or instantiation fails, it warns on stderr and launches the kernels directly; `uses_cuda_graph()` says which happened. Everything else is as in [#15](nsight-profile.md): Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB RAM, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power. TensorRT 11.4.0.106, CUDA 13.4, engines built with `--noTF32` ([#16](tensorrt-runtime.md)), the same engine files as #15 and #16.
- Comparison: `python -m tether.runtime_compare VIDEO --compare cuda-graph --pipeline MODE --model models/<model>.onnx --rounds 2`. It dumps detections without and with the graph (`--dump`, sequential), checks them with #16's agreement tolerance and byte for byte, then runs two timing rounds per side headless in MODE, alternating which goes first. Every frame is timed, including the first.
- Videos: the Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080 H.264, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames), as in #11, #14, #15 and #16. Models: YOLO26n at 640×640 and 1280×736.
- Profiles: the same seven captures as #15, with the graph on, plus one without the graph (Anti-UAV 640, parallel) as a same-session reference. The #15 command, plus `--cuda-graph-trace=node` so Nsight records each kernel the graph runs, not just the graph as a whole:
  ```
  nsys profile -t cuda,nvtx --nvtx-domain-include=default --sample=none --cpuctxsw=none --cuda-graph-trace=node -o RUN \
      tether-pipeline VIDEO --model models/<model>.onnx --runtime tensorrt --pipeline MODE --timing RUN-nsys.jsonl
  ```
  `tools/nsight_steps.py` now also splits a detect that replays a graph: the "launch" part is the `cudaGraphLaunch` call, and the graph's kernels (which all carry that call's correlation id) are added up on the GPU side.
- Files: [`cuda-graph.json`](cuda-graph.json) has all eight comparison results (agreement, every round's timing summary, busy shares from the plain timing logs) and each profile's `nsight_steps.py` result and timing summary. [`cuda-graph/`](cuda-graph/) has the `nsys stats` CSV exports for each capture (`nvtx_sum`, `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`). The raw results, dumps, timing logs and the 4.8–12.1 MB `.nsys-rep` captures are in `results/cuda-graph/` and `results/nsight-graph/` (ignored).

## Same detections

| Video | Input | Checked (≥ 0.25) | With a partner | Dump files |
|---|---|---|---|---|
| Anti-UAV | 640 | 56 | 56 (100%) | byte-identical |
| MOT17-04 | 640 | 30,214 | 30,214 (100%) | byte-identical |
| Anti-UAV | 1280×736 | 4 | 4 (100%) | byte-identical |
| MOT17-04 | 1280×736 | 79,268 | 79,268 (100%) | byte-identical |

Each pair was dumped twice (once per pipeline-mode run of the tool), byte-identical every time. Replaying the same kernels on the same inputs gives the same bits. The unit tests check the same on the brightness stand-in (`tests/onnx/make_models.py`), whose score is the frame's mean value, so a replay that reused an earlier frame's input or output would fail: `CudaGraph.GivesTheSameDetectionsAsLaunchingDirectlyFrameAfterFrame` and `PipelineCliCudaGraph.DumpsTheSameDetectionsWithAndWithoutTheGraph`.

## Timing (no profiler)

Mean ms per frame (P99 in brackets for frame time and detect). "No graph" is `--no-cuda-graph`, "graph" the default.

### Sequential

| Video, input, round | | FPS | Frame time | capture | prep | detect | End-to-end delay |
|---|---|---|---|---|---|---|---|
| Anti-UAV 640, 1 | no graph | 84.14 | 11.89 (17.33) | 2.62 | 3.37 | 5.51 (10.63) | 8.90 |
| | graph | 113.97 | 8.77 (12.64) | 2.51 | 3.33 | 2.56 (6.17) | 5.90 |
| Anti-UAV 640, 2 | graph | 111.81 | 8.94 (12.64) | 2.61 | 3.41 | 2.56 (6.23) | 5.98 |
| | no graph | 81.08 | 12.33 (17.87) | 2.74 | 3.50 | 5.73 (10.57) | 9.24 |
| MOT17-04 640, 1 | no graph | 43.52 | 22.98 (30.13) | 12.23 | 3.60 | 6.37 (11.87) | 10.31 |
| | graph | 50.88 | 19.66 (23.62) | 12.26 | 3.65 | 2.97 (5.91) | 6.98 |
| MOT17-04 640, 2 | graph | 50.73 | 19.71 (23.86) | 12.39 | 3.60 | 2.96 (5.68) | 6.92 |
| | no graph | 44.07 | 22.69 (30.15) | 12.13 | 3.56 | 6.24 (11.40) | 10.14 |
| Anti-UAV 1280×736, 1 | no graph | 55.93 | 17.88 (23.11) | 2.64 | 7.39 | 7.04 (11.80) | 14.45 |
| | graph | 66.27 | 15.09 (20.12) | 2.58 | 7.28 | 4.45 (9.13) | 11.74 |
| Anti-UAV 1280×736, 2 | graph | 65.88 | 15.18 (20.27) | 2.61 | 7.35 | 4.45 (9.17) | 11.81 |
| | no graph | 54.92 | 18.21 (22.92) | 2.66 | 7.47 | 7.31 (11.94) | 14.79 |
| MOT17-04 1280×736, 1 | no graph | 34.90 | 28.65 (35.91) | 12.02 | 7.77 | 7.24 (12.42) | 15.81 |
| | graph | 38.28 | 26.12 (32.48) | 12.09 | 7.82 | 4.57 (9.63) | 13.19 |
| MOT17-04 1280×736, 2 | graph | 37.92 | 26.37 (33.64) | 12.27 | 7.86 | 4.60 (9.52) | 13.26 |
| | no graph | 34.90 | 28.65 (36.62) | 12.08 | 7.69 | 7.25 (13.07) | 15.74 |

### Parallel

Busy share is each step's total time over the run's wall time; the slowest stage is the one near 100%.

| Video, input, round | | FPS | Frame time | capture | prep | detect | Delay |
|---|---|---|---|---|---|---|---|
| Anti-UAV 640, 1 | no graph | 173.43 | 5.77 (9.57) | **5.80, 99.4%** | 3.72, 63.7% | 5.10 (7.42), 87.4% | 9.91 |
| | graph | 191.99 | 5.21 (6.59) | **5.24, 99.5%** | 3.69, 70.1% | 2.32 (2.83), 44.0% | 6.15 |
| Anti-UAV 640, 2 | graph | 191.30 | 5.23 (7.11) | **5.26, 99.5%** | 3.72, 70.4% | 2.35 (2.91), 44.4% | 6.22 |
| | no graph | 172.57 | 5.79 (9.67) | **5.83, 99.4%** | 3.84, 65.4% | 5.16 (7.38), 87.9% | 10.14 |
| MOT17-04 640, 1 | no graph | 66.31 | 15.08 (20.98) | **15.10, 99.8%** | 3.52, 23.3% | 6.03 (11.00), 39.9% | 10.09 |
| | graph | 67.83 | 14.74 (19.56) | **14.75, 99.9%** | 3.56, 24.1% | 2.82 (7.40), 19.1% | 6.85 |
| MOT17-04 640, 2 | graph | 67.50 | 14.82 (19.84) | **14.82, 99.9%** | 3.58, 24.1% | 2.83 (7.23), 19.0% | 6.89 |
| | no graph | 65.89 | 15.18 (21.32) | **15.20, 99.9%** | 3.58, 23.5% | 5.95 (11.10), 39.1% | 10.07 |
| Anti-UAV 1280×736, 1 | no graph | 123.27 | 8.11 (12.00) | 5.92, 72.2% | **8.13, 99.2%** | 6.31 (7.72), 77.1% | 24.89 |
| | graph | 127.96 | 7.81 (8.87) | 5.85, 74.3% | **7.82, 99.3%** | 4.02 (4.64), 51.0% | 21.73 |
| Anti-UAV 1280×736, 2 | graph | 127.39 | 7.85 (8.95) | 5.89, 74.4% | **7.86, 99.2%** | 4.03 (4.66), 50.9% | 21.82 |
| | no graph | 123.17 | 8.12 (11.79) | 5.97, 72.9% | **8.13, 99.2%** | 6.41 (7.91), 78.2% | 24.91 |
| MOT17-04 1280×736, 1 | no graph | 62.98 | 15.88 (20.63) | **15.90, 99.8%** | 7.61, 47.8% | 6.46 (8.15), 40.6% | 15.08 |
| | graph | 65.69 | 15.22 (20.17) | **15.23, 99.9%** | 7.40, 48.5% | 4.29 (8.89), 28.1% | 12.65 |
| MOT17-04 1280×736, 2 | graph | 65.83 | 15.19 (20.11) | **15.21, 99.9%** | 7.34, 48.2% | 4.32 (8.85), 28.4% | 12.63 |
| | no graph | 64.35 | 15.54 (19.64) | **15.56, 99.8%** | 7.44, 47.7% | 6.45 (7.88), 41.4% | 14.90 |

Track was 0.01 ms (Anti-UAV) and 0.33–0.85 ms (MOT17-04) in every run, and draw 0 (headless).

### Speedups

Without-graph time over with-graph time (FPS: with over without), rounds 1 and 2, cut to two decimals.

| Video, input | Sequential detect | Sequential FPS | Parallel detect | Parallel FPS |
|---|---|---|---|---|
| Anti-UAV 640 | 2.15×, 2.24× | 1.35×, 1.37× | 2.19×, 2.19× | 1.10×, 1.10× |
| MOT17-04 640 | 2.14×, 2.11× | 1.16×, 1.15× | 2.13×, 2.10× | 1.02×, 1.02× |
| Anti-UAV 1280×736 | 1.58×, 1.64× | 1.18×, 1.19× | 1.57×, 1.59× | 1.03×, 1.03× |
| MOT17-04 1280×736 | 1.58×, 1.57× | 1.09×, 1.08× | 1.50×, 1.49× | 1.04×, 1.02× |

## Inside detect, with the graph (profiled)

Mean ms per detect call (P99 in brackets), from `tools/nsight_steps.py`. Every detect made exactly one `cudaGraphLaunch`, which ran 217 kernels at 640 and 220 at 1280×736, and no kernel launches of its own. "Copy back (CPU)" is the copy of the output rows into ordinary (pageable) memory, which doesn't return until the GPU has finished, so with the graph it holds the wait for the kernels; "wait" is what's left for `cudaStreamSynchronize`.

| Capture | detect | copy in (CPU) | launch | copy back (CPU) | wait | GPU: copy in | GPU: kernels | GPU idle |
|---|---|---|---|---|---|---|---|---|
| Anti-UAV 640, parallel, 1 | 2.630 (3.444) | 0.473 | 0.407 (0.582) | 1.686 | 0.039 | 0.364 | 1.645 (1.687) | 0.363 |
| Anti-UAV 640, parallel, 2 | 2.661 (3.348) | 0.475 | 0.421 (0.630) | 1.693 | 0.044 | 0.359 | 1.645 (1.690) | 0.378 |
| MOT17-04 640, parallel, 1 | 2.831 (3.591) | 0.538 | 0.438 (0.648) | 1.770 | 0.037 | 0.382 | 1.767 (1.906) | 0.354 |
| MOT17-04 640, parallel, 2 | 2.803 (3.412) | 0.537 | 0.437 (0.668) | 1.747 | 0.036 | 0.380 | 1.764 (1.900) | 0.334 |
| Anti-UAV 640, sequential | 2.714 (3.477) | 0.515 | 0.426 (0.788) | 1.707 | 0.036 | 0.375 | 1.708 (1.869) | 0.338 |
| Anti-UAV 1280×736, parallel | 4.423 (5.363) | 1.133 | 0.535 (0.727) | 2.692 | 0.038 | 0.993 | 2.977 (3.288) | 0.182 |
| MOT17-04 1280×736, parallel | 4.426 (5.328) | 0.991 | 0.430 (0.632) | 2.908 | 0.042 | 0.825 | 3.204 (3.719) | 0.042 |
| *No graph:* Anti-UAV 640, parallel | 6.629 (9.547) | 0.656 | 5.834 (8.759) | 0.073 | 0.054 | 0.585 | 1.477 (1.537) | 4.376 |

The slowest 1% of detect calls (the 11 at or above the P99), mean ms:

| Capture | detect | launch | copy back (CPU) | GPU: kernels | GPU idle |
|---|---|---|---|---|---|
| Anti-UAV 640, parallel, 1 | 3.540 | 0.539 | 2.305 | 1.664 | 1.094 |
| Anti-UAV 640, parallel, 2 | 3.756 | 0.577 | 2.439 | 1.641 | 1.238 |
| MOT17-04 640, parallel, 1 | 4.920 | 1.200 | 2.702 | 1.757 | 2.018 |
| MOT17-04 640, parallel, 2 | 3.866 | 0.633 | 2.231 | 1.801 | 0.973 |
| Anti-UAV 640, sequential | 3.695 | 0.803 | 1.847 | 1.737 | 0.885 |
| Anti-UAV 1280×736, parallel | 5.831 | 0.691 | 3.450 | 2.965 | 1.133 |
| MOT17-04 1280×736, parallel | 5.975 | 0.589 | 3.623 | 3.446 | 0.676 |
| *No graph:* Anti-UAV 640, parallel | 12.421 | 11.507 | 0.088 | 1.471 | 10.065 |

## Reading the numbers

- **The launch time is gone.** The CPU now spends 0.407–0.438 ms (profiled, 640) issuing the whole engine, against 5.834 ms in the same session without the graph and 5.81–6.10 ms in #15. The GPU's idle time inside a detect fell from 4.376 ms to 0.334–0.378 ms at 640, and to 0.042–0.182 ms at 1280×736. What's left of detect is the copy in, the kernels and the copy back: at 640, 0.47–0.54 + 1.65–1.77 + about 0 ms, plus the GPU's 0.33–0.38 ms idle, which is mostly the copy in's GPU time (0.36–0.38 ms) not overlapping with anything.
- **The P99 tail mostly went with it.** In #15 and in this session's no-graph capture, the slowest 1% of detects were slow because launching was slow (11.507 ms of a 12.421 ms tail here). With the graph, the tail at 640 is 3.540–4.920 ms, and its launch part is 0.539–1.200 ms. Without the profiler, detect P99 at 640 went from 7.38–11.87 ms to 2.83–7.40 ms. MOT17-04 in parallel keeps the widest tail (7.23–7.40 ms against a 2.82–2.83 ms mean), and at 1280×736 its detect P99 went *up* (7.88–8.15 → 8.85–8.89 ms) even though the mean fell 1.49–1.50×. The profile can't say why: it points at the copy back (the slowest 1% at 1280×736 spent 3.623 ms there against 2.908 ms on average), not the launch.
- **Kernels read 1.645–1.767 ms with the graph and 1.477 ms without, under the profiler.** That's node-level graph tracing, which records every kernel the graph runs and which Nsight warns can add overhead, or the replayed kernels being slower, or both; these captures can't separate them. The plain runs settle the net effect: detect fell 2.10–2.24× at 640.
- **The pipeline gains as much as detect's share of the slowest path.** Sequentially, on the Anti-UAV clip at 640, detect fell by 2.95–3.17 ms and the frame time by 3.12–3.39 ms (each round's pair), so all of it reached FPS. In parallel mode the frame time is the slowest stage's. That stage was capture at 640 on both videos and MOT17-04 at 1280×736 (busy 99.4–99.9%), and prep on the Anti-UAV clip at 1280×736 (busy 99.2–99.3%), before and after. Those runs gained 1.02–1.10×. On the Anti-UAV clip at 640, capture itself got faster with the graph (5.80–5.83 → 5.24–5.26 ms), which is where its 1.10× came from. A likely cause is the detect thread no longer keeping a CPU core busy issuing launches while capture decodes, but no CPU sampling was done to show it.
- **Delay drops by about what detect lost.** End-to-end delay fell 2.27–3.92 ms in every configuration (each round's pair). It stays highest on the Anti-UAV clip at 1280×736 in parallel (21.73–21.82 ms), where frames queue in front of prep, the slowest stage.
- **Pinned staging for the copies: not built.** The copy in is now 0.47–0.54 ms of CPU and 0.36–0.38 ms of GPU at 640, and 0.99–1.13 / 0.83–0.99 ms at 1280×736: the largest part of detect after the kernels. Copying from pinned memory would likely cut it, and would let the copies run inside the graph. But in parallel mode detect is busy 19.0–51.0% and never the slowest stage, so a faster copy can't raise FPS there. It would only shorten sequential frames and delay, by some part of the 0.47–1.13 ms. That's a hypothesis: pinned copies weren't measured. It also needs a second copy on the CPU (the prepared frame into the pinned buffer), unless #17's CUDA prep writes the input on the GPU directly.

## What this means for #18

With the graph on (the default), TensorRT's detect step without the profiler is:

| Input | Sequential detect | Parallel detect |
|---|---|---|
| 640 | 2.56–2.97 ms (P99 5.68–6.23) | 2.32–2.83 ms (P99 2.83–7.40) |
| 1280×736 | 4.45–4.60 ms (P99 9.13–9.63) | 4.02–4.32 ms (P99 4.64–8.89) |

Detect now grows with input size (1.52–1.73× from 640 to 1280×736, averages of the two rounds), because the kernels, not the launches, are most of it. In parallel mode it's the slowest stage on neither video at either size: capture is (or prep on Anti-UAV at 1280×736). So for the shipped config, parallel FPS is set by decoding at 640 and by prep at 1280×736 on the Anti-UAV clip, which is #17's case.

## Caveats

- One laptop, two rounds per configuration, one profile per configuration (two at 640 parallel), on a desktop that wasn't idle (#14 measured the desktop apps at about two cores' worth of CPU).
- The capture's fallback (an engine TensorRT can't record into a graph) prints a warning and launches directly; no engine here fails capture, so that path is untested. Both YOLO26n engines and both stand-ins capture.
- The graph is captured for one input and output address. `TensorRtDetector` never moves its buffers, so that holds; code that changed them would need to capture again.
- The profiler adds time: profiled detect with the graph is 2.630–2.831 ms at 640 against 2.32–2.83 ms without it.
- Headless only; `--show` and the webcam weren't timed with the graph.
