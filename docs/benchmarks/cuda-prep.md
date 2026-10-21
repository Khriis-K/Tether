# CUDA frame prep: the letterbox done on the GPU and left there for TensorRT (#17)

**Question:** with YOLO26n at 1280×736, the detector #18 recommends, CPU frame prep is the slowest stage in parallel mode on the Anti-UAV clip (busy 99.2–99.3%, 124.82–126.18 FPS). If a CUDA kernel does the prep (resize, BGR to RGB, scale to [0, 1]) and leaves the result on the GPU for TensorRT, how much faster is the prep step, how much time does each frame save, and does it give the same input?

**Answer:** on the headline config (Anti-UAV clip, YOLO26n at 1280×736, parallel, TensorRT with the CUDA graph), **prep is 9.19–9.21× faster** (7.83–7.90 → 0.85–0.86 ms). **FPS goes from 126.63–127.71 to 195.45–195.86 (1.53–1.54×)**, frame time from 7.83–7.90 to 5.11–5.12 ms (**2.72–2.78 ms saved per frame**), P99 frame time from 9.00–9.08 to 6.49–6.57 ms, and end-to-end delay from 21.76–21.95 to 3.74–3.75 ms. Prep stops being the bottleneck: capture (video decoding) is now the slowest stage, busy 99.7%. Sequentially, the same config saves **9.08–9.19 ms per frame** (65.75–66.19 → 166.09–166.34 FPS, 2.50–2.52×).

The model sees **the same input either way**: the detections dumped with CPU and CUDA prep are **byte-identical** on both videos at both sizes. The kernel copies OpenCV's fixed-point resize arithmetic, so its blob is bit-identical to the CPU's on these 1920×1080 frames.

Two other effects, both measured:
- **Detect got faster too, 1.30–1.61×**, because the prepared frame no longer crosses from the CPU: TensorRT's input is filled by a 15.9 µs copy within the GPU instead of a 1.071 ms copy from the CPU (profiled, 1280×736). Detect P99 fell more: on MOT17-04 at 1280×736 in parallel, from 8.79–9.07 to 3.43–3.54 ms.
- **Where prep wasn't the slowest stage, parallel FPS barely moves.** At 640, and on MOT17-04 at either size, capture is the slowest stage before and after (busy 99.4–99.8%): FPS changed 0.98–1.02×. **MOT17-04 at 640 in parallel got slightly slower**, 66.00–66.44 → 64.89–65.13 FPS, because decoding the JPEGs slowed (15.06–15.16 → 15.35–15.40 ms).

The spec's payoff target is "3x+ faster on that step, plus total time saved per frame". It's met in 7 of the 8 configurations (3.06–9.71×), and missed in one: **MOT17-04 at 640, sequential, 2.74–2.78×**.

CUDA prep is a setting: `tether-pipeline --runtime tensorrt --prep cuda`. CPU prep stays the default and the only choice for ONNX Runtime, and builds without the CUDA toolkit have only the CPU path.

## What was built

- **`CudaFramePrep`** (`cpp/src/tether/cuda_frame_prep.{hpp,cu}`), behind the existing `FramePrep` interface. Each `prepare()`:
  1. uploads the 8-bit BGR frame (6,220,800 bytes for 1920×1080) to the GPU on the prep's own CUDA stream, with `cudaMemcpy2DAsync` from the frame's ordinary (pageable) memory, so a cropped frame (rows not back to back) works too;
  2. runs one kernel, `prep_kernel`, with one thread per model-input pixel: if the pixel is in the padding it's gray 114, otherwise a bilinear sample of the frame. Each value is then put into RGB order, laid out channels first, and divided by 255 in float. That's the same output as `CpuFramePrep`;
  3. waits for the stream, so the step's time is the whole prep and the blob is ready for any stream.
  The result's `gpu_blob` holds the blob in GPU memory, and its CPU `blob` is empty. Each frame gets its own buffer from a small pool, returned when the last copy of the prepared frame is dropped. So in parallel mode the next frame can be prepared while TensorRT still reads the one before (`cudaFree` waits for the whole GPU, which is why the buffers are pooled instead of freed).
- **Matching OpenCV's resize, rounding included.** `cv::resize` with `INTER_LINEAR` on 8-bit images weighs the two source pixels on each axis in fixed point (out of 2048), adds across then down in integers, and rounds once. The kernel uses the same source pixels and the same weights, computed on the CPU once per frame size with OpenCV's own formula (`taps()` in the `.cu`). The first version rounded the exact sum, as OpenCV's scalar code does, and differed from the CPU blob by one gray level on 9.1–14.8% of values for 1920×1080 frames into 1280×736 (at 640 every output pixel lands on a source pixel, so nothing is blended and both versions were exact). OpenCV's vectorized vertical pass (`VResizeLinearVec_32s8u`) rounds differently: it drops 4 bits, keeps the high 16 bits of each product and rounds off the last 2. Copying that made 1920×1080 frames bit-identical at both input sizes. Some odd frame sizes still differ by one level on 0.015–0.031% of values (480×360, 517×333 and 640×480 test frames, measured once while building the kernel), likely where OpenCV finishes a row with its scalar code.
- **TensorRT reads the GPU blob where it is.** `TensorRtDetector::detect` copies a GPU blob into the engine's input buffer with a device-to-device `cudaMemcpyAsync` on its own stream, instead of copying the CPU blob up. The CUDA graph (#40) recorded the engine's input address once, so the engine's input buffer can't move. The other way to avoid even this copy is to capture a graph per pooled buffer, or to have prep write into the engine's buffer. Either one ties prep and detect together (parallel mode prepares frame N+1 while detect still reads frame N), so this copy, 15.9 µs per frame profiled at 1280×736, was kept instead. ONNX Runtime rejects a GPU blob with an error saying it needs CPU frame prep.
- **The setting:** `tether-pipeline --prep cpu|cuda` (default `cpu`). `--prep cuda` without `--runtime tensorrt` is a bad argument (exit 2), since ONNX Runtime runs on the CPU. In a build without the CUDA toolkit, `--prep cuda` fails with an error saying to configure CMake with `TensorRT_ROOT`. The stats panel's "frame prep" row reads "CPU" or "CUDA (GPU)". In C++: `enum class Prep`, `make_frame_prep(prep, height, width)`, `cuda_prep_built()`, and `run_settings(..., prep)`.
- **The build:** CMake enables the CUDA language when `TensorRT_ROOT` is set, with `CMAKE_CUDA_ARCHITECTURES 120` (the RTX 5070 Laptop GPU, Blackwell, compute capability 12.0) and nvcc 13.4. With Visual Studio, pick the toolkit with `-T "cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4"` as well as `CUDAToolkit_ROOT`; this machine has CUDA 12.0, 13.1 and 13.4 installed.
- **The comparison tool:** `python -m tether.runtime_compare VIDEO --compare frame-prep --pipeline MODE --model M --rounds 2`, a third comparison next to `runtimes` and `cuda-graph`. It dumps both sides' detections and checks them with #16's agreement tolerance and byte for byte. Then it alternates timing rounds, and its report gives each round's prep speedup and frame time saved.

## Is the CUDA output right?

The tolerance, set in the test before the kernel was written: **every value of the blob within one 8-bit gray level (1/255) of `CpuFramePrep`'s** (plus 1e-6 for float rounding: 101/255 − 100/255 is a hair over 1/255 in float). The tests (`cpp/tests/test_cuda_frame_prep.cpp`, built with TensorRT) cover:
- random-noise frames, the worst case for rounding;
- a drawn scene (shading, filled shapes, a line);
- 1920×1080 into 1280×736 (non-square) and into 640×640;
- a narrow frame enlarged with side padding, odd sizes (1280×5 and 333×517);
- a frame that needs no resize (bit-exact required);
- a cropped frame whose rows aren't back to back;
- the frame size changing between calls;
- the same geometry (scale, padding, sizes) as the CPU prep.

They also check that each prepared frame keeps its own GPU buffer while a later one is prepared. `cpp/tests/test_cuda_prep_detector.cpp` checks that the TensorRT detector gives CPU prep's exact detections from CUDA-prepared frames, with and without the CUDA graph, frame after frame. It uses #40's brightness stand-in model, whose score is the frame's mean, so a replay that read an old frame would show. The same file checks `tether-pipeline --prep cuda --dump` against `--prep cpu` byte for byte, and runs the parallel pipeline with CUDA prep.

On the real model and videos (`tether-runtime-compare --compare frame-prep`, every frame):

| Video | Input | Checked (score ≥ 0.25) | With a partner | Dump files |
|---|---|---|---|---|
| Anti-UAV | 1280×736 | 4 | 4 (100%) | byte-identical |
| MOT17-04 | 1280×736 | 79,268 | 79,268 (100%) | byte-identical |
| Anti-UAV | 640 | 56 | 56 (100%) | byte-identical |
| MOT17-04 | 640 | 30,214 | 30,214 (100%) | byte-identical |

Each pair was dumped twice (once per pipeline-mode run), byte-identical every time. That's expected once the blob is bit-identical: the engine gets the same input.

## Setup

- Code as committed for #17. TensorRT 11.4.0.106 with the CUDA graph on (the default, #40), engines built with `--noTF32` (the same files as #16, #40 and #18), CUDA 13.4, kernel compiled for sm_120. Release build, MSVC 19.44, Windows 11, AMD Ryzen AI 7 350, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power. Headless.
- Comparison: `python -m tether.runtime_compare VIDEO --compare frame-prep --pipeline {parallel,sequential} --model models/{yolo26n_1280x736,yolo26n}.onnx --rounds 2`, so each configuration runs two rounds per side, alternating which side goes first. Every frame is timed, including the first. Busy shares come from the plain timing logs (`busy_share` in `python/tether/model_compare.py`).
- Videos: the Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080 H.264, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames), as in #14–#18.
- Profiles: one Nsight Systems capture per side on the headline config (Anti-UAV, 1280×736, parallel), with #40's command plus `--prep`:
  ```
  nsys profile -t cuda,nvtx --nvtx-domain-include=default --sample=none --cpuctxsw=none --cuda-graph-trace=node -o RUN \
      tether-pipeline VIDEO --model models/yolo26n_1280x736.onnx --runtime tensorrt --prep {cpu,cuda} --pipeline parallel --timing RUN-nsys.jsonl
  ```
- Files: [`cuda-prep.json`](cuda-prep.json) has all eight comparison results (agreement, every round's timing summary with busy shares) and both profiles' `tools/nsight_steps.py` results and timing summaries. [`cuda-prep/`](cuda-prep/) has the `nsys stats` CSV exports for both captures. Raw results, dumps, timing logs and the `.nsys-rep` captures are in `results/cuda-prep/` (ignored).

## Timing (no profiler)

Ranges are min–max over the two rounds. "CPU → CUDA" is `--prep cpu` → `--prep cuda`; P99 in brackets. "Saved per frame" is each round's CPU-prep frame time minus its CUDA-prep frame time.

### Parallel (FPS set by the slowest stage)

| Video, input | prep ms | prep speedup | detect ms (P99) | frame time ms (P99) | saved per frame ms | FPS | FPS gain | end-to-end delay ms | slowest stage (busy) |
|---|---|---|---|---|---|---|---|---|---|
| **Anti-UAV 1280×736** | 7.83–7.90 → 0.85–0.86 | **9.19–9.21×** | 4.09–4.12 (5.10) → 2.78–2.80 (3.33–3.37) | 7.83–7.90 (9.00–9.08) → 5.11–5.12 (6.49–6.57) | **2.72–2.78** | 126.63–127.71 → 195.45–195.86 | 1.53–1.54× | 21.76–21.95 → 3.74–3.75 | prep (99.2%) → capture (99.7%) |
| MOT17-04 1280×736 | 7.37–7.46 → 1.20–1.23 | 6.00–6.21× | 4.30–4.33 (8.79–9.07) → 2.82–2.83 (3.43–3.54) | 15.40–15.53 (20.93–22.18) → 15.23–15.27 (20.00–20.09) | 0.12–0.30 | 64.37–64.94 → 65.46–65.67 | 1.00–1.02× | 12.67–12.79 → 4.99–5.00 | capture (99.8%) → capture (99.8%) |
| Anti-UAV 640 | 3.68–3.74 → 0.83–0.85 | 4.33–4.50× | 2.33–2.34 (2.88–2.94) → 1.79 (2.25–2.30) | 5.21–5.28 (6.46–6.80) → 5.12–5.17 (6.60–6.64) | 0.09–0.11 | 189.44–191.77 → 193.49–195.18 | 1.01–1.02× | 6.16–6.21 → 2.72–2.74 | capture (99.4–99.5%) → capture (99.7%) |
| MOT17-04 640 | 3.63–3.65 → 1.16–1.19 | 3.06–3.13× | 2.78–2.80 (7.25–7.32) → 1.88–1.89 (2.41–2.43) | 15.05–15.15 (19.93–20.00) → 15.35–15.41 (20.10–20.12) | −0.27 to −0.31 (slower) | 66.00–66.44 → 64.89–65.13 | 0.98× | 6.89–6.93 → 3.53–3.54 | capture (99.8%) → capture (99.8%) |

### Sequential (every step adds up)

Here no stage waits on another, so the busy shares are shares of one frame and none is near 100%. The column gives the largest step.

| Video, input | prep ms | prep speedup | detect ms (P99) | frame time ms (P99) | saved per frame ms | FPS | FPS gain | end-to-end delay ms | largest step (share) |
|---|---|---|---|---|---|---|---|---|---|
| **Anti-UAV 1280×736** | 7.32–7.37 → 0.75–0.76 | **9.63–9.71×** | 4.37–4.40 (8.93–9.00) → 2.73 (3.23–3.25) | 15.11–15.21 (20.05–20.12) → 6.01–6.02 (7.10–7.24) | **9.08–9.19** | 65.75–66.19 → 166.09–166.34 | 2.50–2.52× | 11.71–11.78 → 3.49–3.51 | prep (48.2%) → detect (43.7–45.0%) |
| MOT17-04 1280×736 | 7.68–7.78 → 1.42–1.43 | 5.42–5.45× | 4.44–4.45 (8.73–9.17) → 2.76–2.77 (3.27–3.36) | 26.48–26.60 (31.95–32.33) → 17.50–17.60 (22.51–22.72) | 8.98–8.99 | 37.58–37.76 → 56.80–57.14 | 1.51× | 13.06–13.16 → 5.07–5.10 | capture (47.0–47.2%) → capture (70.5%) |
| Anti-UAV 640 | 3.40 → 0.72 | 4.70–4.73× | 2.48 (5.36–5.60) → 1.73 (2.09–2.15) | 8.81–8.83 (12.46–13.09) → 4.84–4.86 (5.87–5.91) | 3.95–3.99 | 113.21–113.45 → 205.91–206.55 | 1.81–1.82× | 5.88–5.89 → 2.46 | prep (38.2%) → capture (48.1–48.3%) |
| MOT17-04 640 | 3.51–3.52 → 1.26–1.28 | 2.74–2.78× | 2.79–2.88 (5.24–5.99) → 1.83–1.84 (2.38) | 19.59–19.60 (22.62–22.84) → 15.83–15.88 (20.19–20.49) | 3.71–3.76 | 51.02–51.04 → 62.98–63.16 | 1.23× | 6.69–6.78 → 3.48–3.49 | capture (63.1–63.6%) → capture (77.4–77.5%) |

The CPU-prep side reproduces #18's numbers for this row (124.82–126.18 FPS, prep busy 99.2–99.3%) and #40's at 640 (191.30–191.99 FPS parallel on the Anti-UAV clip).

## Inside the step (profiled)

One capture per side, Anti-UAV clip, 1280×736, parallel. The profiler adds time, so read these for what's inside a step, and the tables above for how long it takes.

| | CPU prep | CUDA prep |
|---|---|---|
| prep (NVTX range, mean) | 8.09 ms | 0.909 ms |
| bytes copied to the GPU per frame | 11,304,960 (the float blob, in detect) | 6,220,800 (the 8-bit frame, in prep) + an 11,304,960-byte copy within the GPU (in detect) |
| GPU time of the copy up from the CPU, per frame | 1.071 ms | 0.574 ms |
| CPU time of that copy call, per frame | 1.207 ms (in detect) | 0.685 ms (`cudaMemcpy2DAsync`, in prep) |
| `prep_kernel`, per frame | none | 0.046 ms (46.2 µs) |
| copy into TensorRT's input, GPU time | 1.071 ms (from the CPU) | 0.016 ms (15.9 µs, within the GPU) |
| detect (NVTX range, mean) | 4.49 ms | 3.18 ms |
| busy share: capture / prep / detect | 75.2% / 99.1% / 55.0% | 99.6% / 16.0% / 56.1% |

Most of what's left of prep is the upload of the 8-bit frame from ordinary memory: about 0.69 ms of the 0.91 ms profiled. The kernel is 46 µs. The sums are in `cuda-prep/antiuav-1280-*-prep_cuda_gpu_mem_time_sum.csv`, `..._cuda_gpu_kern_sum.csv` and `..._cuda_api_sum.csv`.

## Reading the numbers

- **The prep step is now mostly an upload.** A 1920×1080 frame is 6.2 MB as 8-bit BGR and 11.3 MB as the 1280×736 float blob, so uploading the frame and expanding it on the GPU moves fewer bytes than uploading the blob. At 640 the blob is 4.9 MB, smaller than the frame. That's why the speedup at 640 (2.74–4.73×) is smaller than at 1280×736 (5.42–9.71×): CPU prep there is cheaper, and CUDA prep still uploads the full frame. Uploading from pinned (page-locked) memory might shorten the copy. It wasn't tried; the frame would first need copying into a pinned buffer, since the decoder writes it to ordinary memory.
- **Detect gains 1.30–1.61×, and its tail more.** The copy into TensorRT's input was a 4.9–11.3 MB upload from ordinary memory; now it's a 15.9 µs copy within the GPU. Detect P99 dropped the most where it had been worst: MOT17-04 at 1280×736 in parallel went from 8.79–9.07 to 3.43–3.54 ms. That suggests the upload from pageable memory was much of the P99 tail #40 couldn't explain, though this run doesn't prove it.
- **The parallel pipeline moved to its next bottleneck.** On the headline config, prep's thread is now busy 16.5–16.7% and capture's 99.7%. Capture itself got faster, 5.94–5.98 → 5.13–5.14 ms, likely because prep no longer keeps a CPU core busy beside the decoder. No CPU sampling was done to show it. So 195.45–195.86 FPS is decode-bound, close to the 640 rows (193.49–195.18 FPS with CUDA prep). At 1280×736 the input size no longer costs FPS on this clip.
- **End-to-end delay collapses where prep was the bottleneck.** 21.76–21.95 → 3.74–3.75 ms on the headline config. With prep the slowest stage, every frame waited in the queue in front of it. With capture the slowest, the stages after it are idle when a frame arrives.
- **Where decoding was already the limit, FPS can't move, and on MOT17-04 at 640 in parallel it fell 2%.** Capture there went 15.06–15.16 → 15.35–15.40 ms. A likely cause is the frame upload competing with the JPEG decoder for memory bandwidth or CPU, but that's a hypothesis: no profile was taken of that configuration. On MOT17-04, CUDA prep is also slower and spikier than on the video file (1.16–1.43 ms, P99 5.41–6.00 ms, against 0.72–0.86 ms with P99 1.08–1.57 ms on the Anti-UAV clip). Each JPEG frame is decoded into fresh memory, unlike the video reader. This wasn't investigated.
- **Sequentially, everything adds up.** On the headline config, the frame time fell 9.08–9.19 ms: prep 6.56–6.60 ms, detect 1.64–1.67 ms, and the remaining 0.87–0.92 ms from the other steps and the time between them, which this run doesn't break down. 166.09–166.34 FPS is the first sequential run in this project over the spec's 120 FPS on a video file.

## Caveats

- One laptop, two rounds per configuration, one profile per side, on a desktop that wasn't idle (#14 measured the desktop apps at about two cores' worth of CPU).
- Agreement on the Anti-UAV clip rests on 4 and 56 checked detections; MOT17-04 (79,268 and 30,214) carries it. The blob itself is checked in the unit tests.
- Bit-identical blobs depend on OpenCV choosing its vectorized resize, which depends on the CPU's instruction set. On a CPU where OpenCV fell back to its scalar code, the blobs would differ by one gray level on about one value in ten (the kernel's first version, against this CPU's OpenCV). The unit tests allow that, but the dumps would no longer be byte-identical.
- The kernel was compiled for sm_120 only, so the binary runs on Blackwell GPUs (compute capability 12.x) and nothing older. Setting `CMAKE_CUDA_ARCHITECTURES` adds others.
- Headless only. `--show` and the webcam weren't timed with CUDA prep. CUDA prep is not the default: the shipped default (CPU prep, YOLO26n at 640) is unchanged, and switching it is the user's call, with #18's input size.
- The GPU blobs are pooled and never shrink. A run holds as many as are in flight at once (in parallel mode, up to one per queue slot and stage), 11.3 MB each at 1280×736.
