# The benchmarks reproduced in Docker (#19)

**Question:** does the Docker image (`docker/Dockerfile`, Linux on NVIDIA's TensorRT container) reproduce the committed numbers with one command, and how do its timings compare with the Windows runs?

**Answer:** the one command (`docker run ... tether`, which runs `docker/benchmark.sh`) ran start to finish in **45.7 minutes** and wrote every results file. **Every accuracy number matches the write-ups exactly**: MOT17 HOTA, IDF1 and ID switches for Tether and ByteTrack on both detection sets, the detector's recall and precision, and every Anti-UAV score. The YOLO26n 1280×736 detection files are byte-identical to the Windows ones.

The timings don't match, and they shouldn't be read as the same measurement: different OS, TensorRT 11.3 against 11.4, inside WSL 2. On the Anti-UAV clip with the shipped configuration (YOLO26n at 1280×736, TensorRT, CUDA prep, parallel), the container ran **256.72–259.71 FPS, P99 frame time 5.67–6.17 ms**, against 195.45–195.86 FPS and 6.49–6.57 ms on Windows (`cuda-prep.md`). The bottleneck moved from capture to detect.

## Setup

- The image as committed for #19: `nvcr.io/nvidia/tensorrt:26.09-py3` (digest `sha256:651006ce4ec6…`), TensorRT 11.3.0, CUDA 13.4.1 (nvcc 13.4.59), GCC 13.3.0, Ubuntu 24.04, Python 3.12.3, ONNX Runtime 1.31.0, PyTorch 2.14.1 (CPU), Ultralytics 8.4.181. CUDA frame prep compiled for sm_120 (the run used the sm_120-only build; the default now also covers older GPUs, see the caveats). The built image is 23.6 GB, 19.4 GB of it the TensorRT base image, and building it took about 12 minutes on this laptop, 6 of them OpenCV and the C++ programs.
- The same laptop as the Windows write-ups (AMD Ryzen AI 7 350, NVIDIA GeForce RTX 5070 Laptop GPU, driver 617.42), through Docker Desktop 29.1.3's WSL 2 backend (a Linux VM with 16 CPUs and 15.27 GiB). Not a native Linux machine. Datasets bind-mounted read-only from the Windows drive. Headless.
- The command, from the repo root, with empty `models/docker/` and `results/docker/` folders, so the models were exported and the engine built inside the container:
  ```
  docker run --rm --gpus all -v ./data:/tether/data:ro -v ./models/docker:/tether/models -v ./results/docker:/tether/results tether
  ```
- One full run. Timing is `tether-runtime-compare --compare frame-prep --pipeline parallel --rounds 2`, as in `cuda-prep.md`; busy shares from the timing logs with `busy_share` (`python/tether/model_compare.py`). Results are in `results/docker/` (ignored).

## Accuracy: identical

| | Container | Write-up |
|---|---|---|
| MOT17, Tether, 1280×736 | 41.46 HOTA / 49.38 IDF1 / 732 IDSW | 41.46 / 49.38 / 732 (`antiuav-duplicates.md`, the merge default; `mot17-bytetrack.md` says 731, from before #36's merge) |
| MOT17, ByteTrack, 1280×736 | 40.86 / 48.63 / 800 | 40.86 / 48.63 / 800 |
| MOT17, Tether, 640 | 33.13 / 37.62 / 529 | 33.1 / 37.6 / 529 |
| MOT17, ByteTrack, 640 | 33.46 / 37.91 / 549 | 33.5 / 37.9 / 549 |
| Detector recall / precision, 1280×736 | 54.6% / 84.3% | 54.6% / 84.3% |
| Detector recall / precision, 640 | 33.7% / 85.4% | 33.7% / 85.4% |
| Anti-UAV ID kept, nothing deleted / 20% deleted | 82.6% / 84.0% (53,523 of 63,723) | 82.6% / 84.0% (53,523) (`antiuav-dropout.md`) |
| Anti-UAV gaps recovered, 0.25 / 0.5 / 1 s | 91.5% / 87.9% / 88.9% | 91.5% / 87.9% / 88.9% |

- **The 1280×736 detection files are byte-identical** to the Windows dumps on all 7 MOT17 videos, though the ONNX files themselves differ (near the end of the file, likely export metadata; not checked further).
- **The 640 detection files are not byte-identical:** the boxes match, and scores differ in about the 7th significant digit (0.8675716 against 0.8675720 on the first detection of MOT17-02). Every tracker score still matches. By its file date the Windows 640 dump (2026-10-17) is older than the 1280×736 one (2026-10-18); why it differs wasn't looked into.
- Anti-UAV feeds the tracker labeled boxes, not a detector, so it only tests that the tracker and scoring run the same on Linux.

## Timing: different, and not comparable

Anti-UAV clip `20190925_111757_1_1/visible.mp4` (1920×1080 H.264, 1000 frames), YOLO26n at 1280×736, TensorRT with the CUDA graph, parallel. Ranges are over the two rounds; P99 in brackets. CPU and CUDA prep gave byte-identical detection files here too.

| | CPU prep | CUDA prep | Windows, CUDA prep (`cuda-prep.md`) |
|---|---|---|---|
| FPS | 161.46–163.54 | **256.72–259.71** | 195.45–195.86 |
| frame time ms | 6.11–6.19 (13.25–14.21) | 3.85–3.90 (**5.67–6.17**) | 5.11–5.12 (6.49–6.57) |
| capture ms | 2.33–2.39 | 2.00 | 5.13–5.14 |
| prep ms | 6.01–6.10 | 1.11–1.12 | 0.85–0.86 |
| detect ms (P99) | 5.55 (8.36–10.87) | 3.71–3.77 (5.22–5.49) | 2.78–2.80 (3.33–3.37) |
| end-to-end delay ms | 22.59–22.60 | 17.11–17.40 | 3.74–3.75 |
| slowest stage (busy) | prep (96.5–97.1%) | detect (94.2–94.7%) | capture (99.7%) |

- **CUDA prep: prep 5.37–5.50× faster, FPS 1.588–1.590×, 2.26–2.30 ms saved per frame.** On Windows it was 9.19–9.21×, 1.53–1.54× and 2.72–2.78 ms. The prep speedup is smaller because both sides moved: CPU prep is faster here (6.01–6.10 against 7.83–7.90 ms) and CUDA prep slower (1.11–1.12 against 0.85–0.86 ms).
- **Decoding is 2.57× faster in the container** (2.00 against 5.13–5.14 ms with CUDA prep). The Linux build's OpenCV reads video through Ubuntu's FFmpeg 6 (avcodec 60.31), the Windows one through OpenCV's bundled FFmpeg plugin. That likely explains it, but it wasn't isolated.
- **Detect is slower in the container**, 3.71–3.77 against 2.78–2.80 ms, which makes it the slowest stage. TensorRT 11.3 against 11.4 and running under WSL 2 are both candidates; neither was tested apart.
- **End-to-end delay is higher** (17.11–17.40 against 3.74–3.75 ms), as `cuda-prep.md` predicts when the slowest stage isn't the first one: frames queue in front of detect.

## Caveats

- One run on one laptop, through WSL 2. A native Linux machine, or another GPU, may time differently; accuracy shouldn't move, since the detections are the same bytes.
- The newest TensorRT container (26.09) has TensorRT 11.3.0; the Windows numbers used 11.4.0.106.
- Only the Anti-UAV clip was timed. MOT17-04 (JPEG files), the sequential pipeline and 640 weren't run in the container.
- The timed run's CUDA prep was built for sm_120 only. The image's default is now every architecture NVIDIA's container supports (compute capability 7.5 to 12.0), so the one command also runs on older GPUs; that build passed `ctest` on this laptop's sm_120 GPU, but no other GPU has run it.
