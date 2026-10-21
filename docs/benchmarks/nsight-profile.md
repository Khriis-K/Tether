# Where frame time goes: an Nsight Systems profile of the parallel pipeline (#15)

**Question:** in the parallel C++ pipeline, with TensorRT running the detector (#16), how much of each frame does each step take, and is CPU frame prep a big enough bottleneck to be worth a CUDA kernel (#17)?

**Answer:** **no-go on #17 at the shipped detector config (YOLO26n at 640).** In parallel mode, FPS is set by the slowest stage, and at 640 prep is never it. Prep takes 3.53–3.80 ms a frame and its thread is busy 64.2–64.7% of the run on the Anti-UAV clip and 22.2–23.3% on MOT17-04. Capture's thread (video or JPEG decode) is busy 99.2–99.8%. Taking prep off the CPU wouldn't speed up the stage that sets FPS, unless the freed CPU sped up decoding. Even then, on the Anti-UAV clip detect (5.10–5.18 ms) sits right behind capture. The time worth chasing is somewhere else. Profiled at 640, TensorRT's detect is 6.51–6.73 ms in parallel runs (5.10–6.05 ms without the profiler), but the GPU computes for only 1.47–1.53 ms of it: the CPU spends 5.81–6.00 ms launching the engine's 217 kernels one at a time, and the GPU sits idle 4.29–4.49 ms per frame waiting for them. That launch time is also why TensorRT's detect P99 is 1.43–1.91× its mean in parallel runs at 640. `trtexec` on the same engine needs 4.04–4.06 ms to enqueue without a CUDA graph and 0.028–0.029 ms with one. **#17 becomes a go if #18 ships a 1280×736 (or larger) input.** At 1280×736, prep (7.88–8.21 ms) is the slowest stage on the Anti-UAV clip, its thread busy 99.1%, and removing it would raise that clip's ceiling 1.26–1.28×. MOT17-04 stays capped by JPEG decode at either size.

Along the way: parallel + TensorRT at 640 runs the Anti-UAV clip at **172.30–173.76 FPS with a 9.37–9.60 ms P99 frame time**, past the spec's 120 FPS and 16 ms targets. MOT17-04 runs at **63.20–65.51 FPS, P99 21.84–23.02 ms**, missing both, because decoding its JPEGs takes 15.28–15.83 ms a frame.

## Setup

- Code: the parallel pipeline (`run_parallel`, #14) with TensorRT (#16), plus NVTX ranges added for this issue. `StepRange` in `cpp/src/tether/pipeline.cpp` names each step (capture, prep, detect, track, draw) on the profiler's timeline. It's built only with TensorRT, which brings the CUDA toolkit and its header-only NVTX. NVTX does nothing unless a profiler is attached. Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350 (8 cores, 16 threads), 31 GB RAM, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power. TensorRT 11.4.0.106, CUDA 13.4, engines built with `--noTF32` as in [#16](tensorrt-runtime.md). ONNX Runtime 1.31.0, CPU provider, default threads.
- Profiler: Nsight Systems 2026.3.2, CLI only. Command, per run:
  ```
  nsys profile -t cuda,nvtx --nvtx-domain-include=default --sample=none --cpuctxsw=none -o RUN \
      tether-pipeline VIDEO --model models/yolo26n.onnx --runtime tensorrt --pipeline parallel --timing RUN-nsys.jsonl
  ```
  `--nvtx-domain-include=default` keeps the pipeline's own ranges and drops TensorRT's per-layer ranges. A 100-frame trial with those layer ranges on recorded 27,383 NVTX events. CPU sampling and context-switch tracing are off; on Windows they need an administrator.
- **Plain runs:** each configuration was also run without the profiler (the same command, no `nsys`). Two rounds each, alternating which goes first. Step times, FPS, delay and busy shares below come from the plain runs. The profile adds 0.67–1.49 ms to TensorRT's detect step (Anti-UAV 640: 5.10→6.60 ms and 5.18→6.62 ms; MOT17-04: 5.78→6.50 ms and 6.05→6.72 ms). On the Anti-UAV clip that's enough to make detect the slowest stage under the profiler when capture is the slowest without it. So the profile is for *what's inside* a step, and the plain runs are for *how long* it takes.
- Videos: the Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080 H.264, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames), as in #11, #14 and #16. Headless, every frame timed, including the first.
- Analysis: `tools/nsight_steps.py` reads the profile's SQLite export (`nsys export --type sqlite`). For each detect range, it takes the CUDA calls made on the detect thread inside it and splits the detect into: the copy to the GPU (`cudaMemcpyAsync`, host side), "launch" (from that call returning to the copy back being called, which is `enqueueV3` launching the kernels), the copy back, the wait (`cudaStreamSynchronize`), and the rest (decoding, which is negligible). On the GPU side it adds up the copies and the kernels those calls started. "GPU idle" is the time between the first copy starting and the last one ending when the GPU had nothing to run. **Busy share** is a step's total time over the run's wall time. In parallel mode each stage has its own thread, so the slowest stage is the one near 100%.
- Files: [`nsight-profile.json`](nsight-profile.json) has every run's timing-log summary, busy shares and `nsight_steps.py` result. [`nsight-profile/`](nsight-profile/) has the `nsys stats` CSV exports for each capture (`nvtx_sum`; for TensorRT also `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`) and the four `trtexec` logs. The `.nsys-rep` captures aren't committed: TensorRT's are 12.0–12.7 MB each. They and the timing logs are in `results/nsight/` (ignored). Rerun the command above to get them back. No GUI screenshots were taken; everything here comes from the CLI exports.

## Time per step (no profiler)

Parallel pipeline, mean ms per frame (P99 in brackets), and each step's busy share. FPS and frame time are set by the slowest stage (in bold).

| Video, input, runtime, round | FPS | Frame time | Delay | capture | prep | detect | track |
|---|---|---|---|---|---|---|---|
| Anti-UAV 640, TRT, 1 | 173.76 | 5.75 (9.37) | 10.35 | **5.78 (8.96), 99.2%** | 3.74 (6.96), 64.2% | 5.10 (7.32), 87.6% | 0.01, 0.1% |
| Anti-UAV 640, TRT, 2 | 172.30 | 5.80 (9.60) | 9.92 | **5.84 (8.89), 99.4%** | 3.80 (7.37), 64.7% | 5.18 (7.83), 88.3% | 0.01, 0.2% |
| MOT17-04 640, TRT, 1 | 63.20 | 15.82 (21.84) | 9.83 | **15.83 (20.35), 99.8%** | 3.53 (4.16), 22.2% | 5.78 (10.84), 36.4% | 0.36, 2.3% |
| MOT17-04 640, TRT, 2 | 65.51 | 15.26 (23.02) | 10.18 | **15.28 (21.88), 99.8%** | 3.56 (4.40), 23.3% | 6.05 (11.61), 39.5% | 0.40, 2.6% |
| Anti-UAV 640, ORT, 1 | 29.78 | 33.57 (40.15) | 161.65 | 6.04 (8.30), 17.9% | 4.33 (5.43), 12.8% | **33.55 (40.12), 99.7%** | 0.01, 0.0% |
| Anti-UAV 640, ORT, 2 | 29.95 | 33.38 (39.12) | 160.67 | 6.04 (7.46), 18.0% | 4.28 (5.76), 12.8% | **33.36 (38.79), 99.7%** | 0.01, 0.0% |
| MOT17-04 640, ORT, 1 | 30.26 | 33.03 (38.04) | 149.24 | 16.25 (19.11), 49.0% | 4.08 (4.95), 12.3% | **33.04 (38.03), 99.8%** | 0.45, 1.3% |
| MOT17-04 640, ORT, 2 | 29.37 | 34.04 (55.40) | 154.12 | 16.48 (22.34), 48.3% | 4.13 (6.66), 12.1% | **34.04 (56.04), 99.7%** | 0.46, 1.3% |
| Anti-UAV 1280×736, TRT, 1 | 126.89 | 7.88 (11.77) | 24.22 | 5.90 (11.38), 74.2% | **7.88 (10.86), 99.1%** | 6.24 (9.07), 78.5% | 0.01, 0.1% |
| Anti-UAV 1280×736, TRT, 2 | 121.95 | 8.19 (12.08) | 25.08 | 6.02 (10.26), 72.7% | **8.21 (11.42), 99.1%** | 6.37 (8.07), 76.9% | 0.01, 0.1% |
| MOT17-04 1280×736, TRT, 1 | 63.13 | 15.83 (20.99) | 15.20 | **15.86 (19.67), 99.8%** | 7.50 (8.53), 47.2% | 6.66 (11.40), 41.9% | 0.87, 5.5% |
| MOT17-04 1280×736, TRT, 2 | 64.09 | 15.60 (20.99) | 15.53 | **15.62 (19.43), 99.8%** | 7.54 (8.54), 48.2% | 6.94 (11.98), 44.4% | 0.86, 5.5% |

"Delay" is the mean end-to-end delay (capture start to draw end). Draw is headless, so it's 0.00 ms throughout. TRT is TensorRT, ORT is ONNX Runtime (CPU).

For reference, one sequential TensorRT run on the Anti-UAV clip at 640 (no profiler): 74.85 FPS, a 13.35 ms frame of capture 3.31, prep 3.79, detect 5.75 (P99 11.27) and track 0.01 ms. **Prep is 28.3% of a sequential frame** there, against 24.7% for capture and 43.0% for detect.

### Frame prep's share

- **Parallel, 640, TensorRT:** prep is 3.53–3.80 ms, busy 22.2–64.7%, never the slowest stage. Its time equals 65.0–65.5% of the frame time on the Anti-UAV clip and 22.3–23.3% on MOT17-04, but in parallel mode that's time on its own thread, overlapped with the other steps, not time added to the frame.
- **Parallel, 640, ONNX Runtime:** prep is 4.08–4.33 ms, busy 12.1–12.8%. The 33.04–34.04 ms detector hides it completely.
- **Parallel, 1280×736, TensorRT:** prep is 7.50–8.21 ms. On the Anti-UAV clip it's the slowest stage (busy 99.1%). On MOT17-04 it's busy 47.2–48.2%, behind capture.
- **Sequential, 640, TensorRT:** 28.3% of the frame (one run).

## Inside TensorRT's detect step (profiled)

Mean ms per detect call (P99 in brackets), from `tools/nsight_steps.py` on each capture. Each detect launched exactly 217 kernels at 640 and 220 at 1280×736.

| Capture | detect | copy in (CPU) | launch | copy back (CPU) | wait | GPU: copy in | GPU: kernels | GPU idle |
|---|---|---|---|---|---|---|---|---|
| Anti-UAV 640, parallel, 1 | 6.610 (9.427) | 0.623 | 5.845 (8.240) | 0.075 | 0.055 | 0.555 | 1.470 (1.530) | 4.394 |
| Anti-UAV 640, parallel, 2 | 6.624 (9.352) | 0.626 | 5.857 (8.533) | 0.072 | 0.055 | 0.558 | 1.473 (1.543) | 4.401 |
| MOT17-04 640, parallel, 1 | 6.510 (10.648) | 0.493 | 5.819 (9.878) | 0.096 | 0.063 | 0.369 | 1.538 (1.639) | 4.295 |
| MOT17-04 640, parallel, 2 | 6.731 (9.955) | 0.513 | 6.001 (8.978) | 0.102 | 0.068 | 0.369 | 1.531 (1.625) | 4.491 |
| Anti-UAV 640, sequential | 6.810 (13.503) | 0.522 | 6.101 (12.774) | 0.095 | 0.059 | 0.379 | 1.533 (1.621) | 4.584 |
| Anti-UAV 1280×736, parallel | 7.444 (10.714) | 1.219 | 6.034 (9.124) | 0.113 | 0.056 | 1.109 | 2.583 (2.758) | 3.501 |
| MOT17-04 1280×736, parallel | 7.412 (9.762) | 0.999 | 6.176 (8.114) | 0.119 | 0.060 | 0.830 | 2.706 (3.099) | 3.510 |

And the slowest 1% of detect calls (the 11 at or above the P99), mean ms:

| Capture | detect | launch | GPU: kernels | GPU idle |
|---|---|---|---|---|
| Anti-UAV 640, parallel, 1 | 11.480 | 10.096 | 1.483 | 8.929 |
| Anti-UAV 640, parallel, 2 | 11.018 | 9.953 | 1.508 | 8.483 |
| MOT17-04 640, parallel, 1 | 12.864 | 11.983 | 1.521 | 10.500 |
| MOT17-04 640, parallel, 2 | 11.280 | 10.215 | 1.523 | 8.755 |
| Anti-UAV 640, sequential | 16.726 | 15.831 | 1.508 | 14.376 |
| Anti-UAV 1280×736, parallel | 12.492 | 10.812 | 2.535 | 8.348 |
| MOT17-04 1280×736, parallel | 12.091 | 9.636 | 2.636 | 7.887 |

`trtexec` on the same 640 engine, 1000 runs after 500 warm-up, two rounds (`nsight-profile/trtexec-*.log`). TensorRT 11's `trtexec` captures a CUDA graph by default, and it copies nothing to or from the host.

| `trtexec` | Throughput | Latency mean (P99) | Enqueue mean (P99) |
|---|---|---|---|
| CUDA graph (default), 1 | 543.53 qps | 1.838 (2.260) ms | 0.029 (0.144) ms |
| CUDA graph (default), 2 | 543.28 qps | 1.839 (2.334) ms | 0.028 (0.180) ms |
| `--noCudaGraph`, 1 | 241.68 qps | 4.124 (7.912) ms | 4.066 (7.875) ms |
| `--noCudaGraph`, 2 | 242.27 qps | 4.116 (7.415) ms | 4.049 (7.403) ms |

Without a graph, `trtexec` warns that throughput "may be bound by Enqueue Time rather than GPU Compute".

## Reading the numbers

- **TensorRT's detect is launch-bound, not compute-bound.** Across all five 640 captures (four parallel, one sequential), of a 6.51–6.81 ms profiled detect, the GPU computes for 1.47–1.53 ms. The CPU spends 5.81–6.10 ms in `enqueueV3` launching 217 kernels, 26.8–28.1 µs each under the profiler, and the GPU idles 4.29–4.58 ms between them. The plain runs' detect is 5.10–6.05 ms in parallel and 5.75 ms sequentially, so the profiler adds some of that launch time but not most of it. `trtexec --noCudaGraph` agrees independently: 4.04–4.06 ms to enqueue with no host copies at all. This answers #16's "detect barely grows with input size": at 1280×736 the kernels take 2.58–2.70 ms instead of 1.47–1.53 ms, but the launches (220 of them) cost about the same, 6.03–6.17 ms.
- **This is also where TensorRT's P99 comes from.** In the slowest 1% of detect calls at 640, kernel time is unchanged (1.48–1.52 ms) and launch time doubles (9.95–11.98 ms parallel, 15.83 ms sequential). That's the CPU side being slow to issue launches, not the GPU being slow. It happens sequentially too, where no other pipeline stage is running, so it isn't the stages fighting each other. With a CUDA graph, `trtexec`'s P99 latency is 1.22–1.26× its mean, against 1.80–1.91× without one. Why individual launches stall (Windows' GPU driver model, CPU power states, the profiler) can't be told from this capture: CPU sampling and context-switch tracing were off.
- **The copy to the GPU is small.** The 4.91 MB float32 input at 640 (4,915,200 bytes) takes 0.36–0.55 ms on the GPU and 0.49–0.62 ms of CPU time, from ordinary (pageable) memory. At 1280×736 the 11.30 MB input takes 0.83–1.10 ms on the GPU.
- **At 640, capture is the wall, not prep.** Decoding is the slowest stage in all four TensorRT runs at 640. The H.264 clip decodes in 5.78–5.84 ms in parallel (3.31 ms sequentially, so overlapping steps slow it, as #14 saw). MOT17-04's JPEGs take 15.28–15.83 ms. Detect (5.10–6.05 ms) is second on both videos and prep (3.53–3.80 ms) third.
- **Delay stays low when the first stage is the slowest.** With TensorRT, end-to-end delay in parallel mode is 9.83–10.35 ms at 640, about the sequential 9.55 ms, against 149.24–161.65 ms with ONNX Runtime. When capture is the bottleneck, no frames pile up in the queues after it ([Little's law](../CONCEPTS.md#littles-law)). That's also why the profiled Anti-UAV runs show 26.96–27.03 ms: the profiler makes detect the slowest stage, and frames queue in front of it again.

## Go / no-go on #17 (CUDA frame prep)

**No-go at the shipped config (YOLO26n at 640).** Reasons, in order of weight:

1. **Prep isn't the bottleneck, so a faster prep can't raise FPS.** Parallel mode is the pipeline's fast mode, and its frame time is the slowest stage's time. At 640 with TensorRT that's capture on both videos (busy 99.2–99.8%). Prep is busy 22.2–64.7%. A prep step taking zero time would leave the frame time at capture's, 5.78–5.84 ms on Anti-UAV and 15.28–15.83 ms on MOT17-04, unless freeing a CPU core made decoding faster. That's possible: capture takes 3.31 ms sequentially against 5.78–5.84 ms in parallel, so the overlapping steps do slow it. But these runs can't say how much of that is prep. And on the Anti-UAV clip detect (5.10–5.18 ms) is right behind, so a faster capture there could gain at most 1.11–1.12× (frame time over detect time) before detect sets the pace.
2. **The bigger, cheaper win is in detect.** Of TensorRT's 5.10–6.05 ms detect, roughly 4 ms is the CPU issuing kernel launches while the GPU waits (4.29–4.49 ms GPU idle profiled; 4.04–4.06 ms enqueue in `trtexec --noCudaGraph`). TensorRT can replay a whole engine from a captured CUDA graph, which took `trtexec`'s enqueue from 4.04–4.06 ms to 0.028–0.029 ms and its P99 latency from 1.80–1.91× its mean to 1.22–1.26×. In the pipeline that's a hypothesis, not a measurement: no CUDA graph was tried there, and `trtexec`'s 1.83 ms is the engine alone, without the pipeline's copies. It's likely a small change in `TensorRtDetector` (capture `enqueueV3` once, then launch the graph), not a new kernel.
3. **At 640, a CUDA prep would copy *more* bytes to the GPU, not fewer.** It uploads the raw 1920×1080 BGR frame, 6.22 MB of uint8 (6,220,800 bytes), instead of today's 4.91 MB float32 input. The current copy costs 0.36–0.55 ms of GPU time.
4. **Delay doesn't need it.** Prep is 28.3% of a sequential frame (3.79 of 13.35 ms), so a CUDA prep would cut sequential frame time and delay by up to about that, minus its own cost. But parallel delay with TensorRT is already 9.83–10.35 ms on video files. The spec's 50 ms target is for the webcam, which hasn't been timed with TensorRT, so this is only a sign that delay isn't where the problem is.

**When it becomes a go:** if #18 picks a 1280×736 (or larger) input as the shipped detector. At 1280×736, prep (7.88–8.21 ms) is the slowest stage on the Anti-UAV clip (busy 99.1%), and the next stage (detect, 6.24–6.37 ms) sets a ceiling 1.26–1.28× higher (156.8–160.0 FPS). At that size CUDA prep also shrinks the upload, 6.22 MB of raw frame against 11.30 MB of float32 input (11,304,960 bytes). MOT17-04 is capped by JPEG decode at both sizes (63.13–65.51 FPS), so CUDA prep wouldn't help it at either. #17's own note says the same: "the payoff of this kernel may grow if the shipped input size goes up".

What would raise FPS at 640, in order: a CUDA graph for TensorRT's enqueue (detect), then a faster capture. That means decoding JPEG sequences or video off the CPU, or on more than one thread. Neither has an issue yet. The spec lists CUDA frame prep as part of Stage 4, so building #17 anyway as a learning exercise is a scope call for the user. This profile says it won't make the shipped config faster.

## Caveats

- One laptop, two rounds per configuration (one for the sequential and 1280×736 profiles), on a desktop that wasn't idle (#14 measured the desktop apps at about two cores' worth of CPU). Round-to-round differences are up to 4.93 FPS (Anti-UAV at 1280×736: 126.89 against 121.95) and 2.30 FPS at 640 (MOT17-04).
- The profiler adds 0.67–1.49 ms to TensorRT's detect step, so the detect split is in profiled time. The plain runs' detect is shorter, and the launch share is a little smaller than the profile shows.
- Busy share counts a step's own time. It doesn't show what a step waits on inside itself (a decode slowed by memory traffic from the other threads, say). CPU sampling would, and it needs administrator rights.
- Only 640 and 1280×736 with YOLO26n were profiled, and headless only: with `--show`, draw cost 10.16–13.88 ms a frame in #13's runs ([stats-panel.md](stats-panel.md)), more than any step here at 640 with TensorRT, so it would likely set the pace there (not measured).
