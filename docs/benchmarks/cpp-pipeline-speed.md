# C++ vs. Python: the whole sequential pipeline (#11)

**Question:** with the C++ pipeline built (video source, frame prep, ONNX Runtime detector, tracker), how much faster is the whole thing than the Python one on the same video? Do the two do the same work? And does webcam mode keep up with the camera?

**Answer:** the whole pipeline is **1.08–1.20× faster** in C++ (FPS 22.0–23.3 → 25.1 on an Anti-UAV clip, 16.7 → 19.6 on MOT17-04 at 640, 8.6 → 10.4 at 1280×736). That's far below the tracking step's 14–35× ([#10](cpp-tracking-speed.md)) because the detector takes 66–83% of every C++ frame, and it's the same ONNX Runtime CPU call in both languages ([Amdahl's law](../CONCEPTS.md#amdahls-law)). The spec's "whole pipeline 2x+" can't be reached while both sides run the same CPU detector: even a C++ pipeline whose other steps took no time at all would be at most 1.27–1.35× (Anti-UAV), 1.76× (MOT17-04 at 640) or 1.56× (1280×736) faster. Both pipelines write **byte-identical detection files**. On the laptop webcam the C++ pipeline **doesn't keep up**: 24.4 FPS against a 30 FPS camera, with 69 frames dropped in 300.

## Setup

- Code: the working tree committed as a83d199. The comparison runs were made just before one last fix to `tether-pipeline-compare` (it now sets `PYTHONPATH` itself so the Python side runs this checkout's code; for these runs it was set by hand), which doesn't change what's timed.
- Machine: Windows 11, AMD Ryzen AI 7 350, 31 GB RAM, on mains power. C++: Release, MSVC 19.44. Python 3.14.0, NumPy 2.3.4. Both sides use ONNX Runtime 1.31.0 (CPU provider, default threads) and OpenCV 5.0.0: the C++ build pins the same versions the Python environment has.
- Command: `python -m tether.pipeline_compare VIDEO [--model M]`. It runs `tether-demo --headless` and then `tether-pipeline` on the same file, one after the other, each in its own process. Neither draws (the C++ pipeline is headless until #12), so `draw` is about 0 ms on both sides. Every frame of the video is timed, including warm-up. One run per pipeline per video, except the Anti-UAV clip, which ran twice.
- Videos: Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080, MPEG-4, 1000 frames) and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames). Models: YOLO26n at 640×640 (`models/yolo26n.onnx`, the default) and at 1280×736 (ADR 0002's benchmark size).
- Results files: `results/pipeline-compare/{antiuav-640-run1, antiuav-640-run2, mot17-04-640, mot17-04-1280x736}.json`, each with both timing logs next to it (not committed; `results/` is ignored).

## Results

Mean milliseconds per frame. Speedup is Python's time over C++'s.

| Video, model input | | Frame time | P99 frame time | End-to-end delay | capture | prep | detect | track | FPS |
|---|---|---|---|---|---|---|---|---|---|
| Anti-UAV clip, 640 (run 1) | Python | 45.56 | 52.46 | 39.94 | 5.71 | 5.87 | 33.68 | 0.29 | 22.0 |
| | C++ | 39.84 | 44.38 | 36.60 | 2.77 | 3.55 | 33.04 | 0.01 | 25.1 |
| | speedup | **1.14×** | | 1.09× | 2.1× | 1.7× | 1.02× | 25× | |
| Anti-UAV clip, 640 (run 2) | Python | 42.90 | 47.33 | 38.71 | 4.21 | 4.62 | 33.70 | 0.29 | 23.3 |
| | C++ | 39.86 | 43.62 | 36.65 | 2.76 | 3.53 | 33.11 | 0.01 | 25.1 |
| | speedup | **1.08×** | | 1.06× | 1.5× | 1.3× | 1.02× | 25× | |
| MOT17-04, 640 | Python | 59.71 | 64.74 | 42.81 | 16.90 | 4.86 | 33.84 | 3.99 | 16.7 |
| | C++ | 50.98 | 58.70 | 37.57 | 12.95 | 3.74 | 33.45 | 0.38 | 19.6 |
| | speedup | **1.17×** | | 1.14× | 1.3× | 1.3× | 1.01× | 10.5× | |
| MOT17-04, 1280×736 | Python | 115.67 | 125.08 | 100.12 | 15.49 | 11.96 | 74.14 | 13.91 | 8.6 |
| | C++ | 96.21 | 101.48 | 82.79 | 12.46 | 7.80 | 74.09 | 0.90 | 10.4 |
| | speedup | **1.20×** | | 1.21× | 1.2× | 1.5× | 1.00× | 15.4× | |

**Same work on both sides.** `tether-pipeline VIDEO --dump` and `tether-demo VIDEO --dump` wrote byte-identical detection files on the Anti-UAV clip (1000 frames, 452 detections, 640 model) and on MOT17-09 (525 frames, 9,485 detections, 1280×736 model). Frame prep matches Python's to the bit: the same OpenCV resize, Python's round-half-to-even for the resized size, float32 scaling, and box decoding in float32 as NumPy does it. So the speedups compare the same computation, not a cheaper one.

**Webcam (the laptop's built-in camera, 640×480 at 30 FPS, 640 model, 300 frames):** 24.4 FPS, frame time 40.92 ms (P99 63.50), detect 33.65 ms, **69 frames dropped**. The count comes from gaps in the camera's own frame timestamps. Checking it independently: the 300 frames span 12,257 ms, which is 367.7 camera periods, so about 69 frames went missing, the same as counted. The drops come in threes about every 13 frames. That fits a driver buffer that fills while the pipeline falls behind, then drops the frames it can't hold.

## Reading the numbers

- **The detector sets the pace.** `detect` is within 2% between the languages (it's the same ONNX Runtime call on the same blob) and takes 66% (MOT17-04, 640) to 83% (Anti-UAV) of a C++ frame. C++ wins only on the rest: decoding (`capture`, 1.2–2.1×), frame prep (1.3–1.7×) and tracking (10–25×). On Anti-UAV, tracking was already 0.3 ms in Python, so its 25× buys almost nothing.
- **The 2x+ target needs a faster detector, not a faster language.** The bound in the answer is Python's frame time over Python's detect time: that's the speedup if everything but detection were free. TensorRT and CUDA frame prep (later issues) are what can move it.
- **Run-to-run noise is mostly on the Python side.** The two Anti-UAV runs differ by 6% in Python frame time (45.56 vs. 42.90 ms) and by 0.05% in C++ (39.84 vs. 39.86). Both differences are mostly `capture` and `prep`, so read the speedup's second digit loosely.
- **Webcam mode can't keep up yet.** With a 33.3 ms camera period and a 33.7 ms mean detect step, the sequential C++ pipeline is slower than the camera before counting anything else. The spec's "zero dropped frames" needs the parallel pipeline (#14) and a faster detector, and this is the baseline it has to beat.

## Caveats

- One machine, one run per video (two for Anti-UAV), every frame including warm-up. The two pipelines ran one after the other, not interleaved, so a thermal or power change between them would land on one side.
- `capture` for MOT17-04 is JPEG decoding through OpenCV's image-sequence reader, not video decoding. The two are different code paths in both languages.
- The C++ pipeline was built and run on Windows only. The CMake build has a Linux branch for ONNX Runtime that hasn't been tried.
- The webcam result is one 300-frame run on one camera. How many frames a camera drops depends on its driver's buffering.
