# The stats panel: do its numbers match the log, and what does the window cost? (#13)

**Question:** the `--show` window now has a Dear ImGui stats panel beside the video. Do its live numbers agree with `tether-timing` on the same run's timing log? And how much does drawing the window slow the pipeline down?

**Answer:** the panel's numbers **match the log** to the panel's one decimal. At the end of a 400-frame MOT17-09 run it showed 13.3 FPS, 75.2 ms average and 115.4 ms P99 frame time, 58.4 ms end-to-end delay, and step means of 15.7 / 5.2 / 39.7 / 0.3 / 13.2 ms. `tether-timing`'s summary of the last 300 frames of that run's log gives the same values to the panel's one decimal. Drawing the window costs **10.2–13.9 ms a frame** on average: that's the `draw` step's mean over two `--show` runs of the same 400 frames, against 0.00 ms headless. The FPS change is harder to pin down: 16.1 headless, then 13.3 and 16.6 with the window, because the whole laptop ran faster during the second window run (its `detect` averaged 33.0 ms against 39.5). Treat the cost as a rough size, not a precise figure.

## Setup

- Code: commit 85cd84c (the panel; later review fixes only reorganized drawing code and docs). Release build, MSVC, on the same Windows 11 laptop as [the pipeline write-up](cpp-pipeline-speed.md) (AMD Ryzen AI 7 350).
- Video: MOT17-09 (`img1/%06d.jpg`, 1920×1080 JPEGs), the first 400 frames, with YOLO26n at 640 (`models/yolo26n.onnx`) on ONNX Runtime 1.31.0 (CPU).
- Window run: `tether-pipeline … --show --max-frames 400 --screenshot panel.png`. The window came up at 2432×994 pixels on this machine's display, including the 360-point panel at the monitor's content scale.
- Headless run: the same command without `--show` or `--screenshot`, right after.
- Second window run: the same `--show` command again, after the review fixes (which only reorganized drawing code), about 20 minutes later. Its panel showed the same layout.
- Timing logs and summaries: kept outside the repo (`results/` is ignored). The screenshot is `docs/media/panel-mot17-09.jpg`.

## Results

### The panel against the log

The panel covers the most recent 300 frames ([`LiveStats`](../../cpp/src/tether/live_stats.hpp)), so the check summarizes the log's last 300 lines with `tether.timing.summarize`, the function behind `tether-timing`. The screenshot is taken after the last frame's times have been added.

| | Panel | `tether-timing`, last 300 frames |
|---|---|---|
| FPS | 13.3 | 13.3 |
| Frame time, mean / P99 | 75.2 / 115.4 ms | 75.2 / 115.4 ms |
| End-to-end delay, mean | 58.4 ms | 58.4 ms |
| capture / prep / detect / track / draw | 15.7 / 5.2 / 39.7 / 0.3 / 13.2 ms | 15.7 / 5.2 / 39.7 / 0.3 / 13.2 ms |

They agree because they read the same `StepTimes` the timing log is written from, and because the C++ P99 is the same [nearest-rank](../CONCEPTS.md#nearest-rank-percentile) rule as the Python one. A unit test pins the C++ `p99` to nearest-rank values.

### The cost of the window

Whole-run summaries (all 400 frames) from `tether-timing`, mean / P99 in ms:

| | Frame time | End-to-end delay | capture | prep | detect | track | draw | FPS |
|---|---|---|---|---|---|---|---|---|
| Headless | 62.12 / 75.53 | 45.28 / 55.15 | 15.93 | 5.42 | 39.53 | 0.34 | 0.00 | 16.1 |
| `--show`, run 1 | 74.93 / 115.36 | 59.02 / 93.68 | 15.61 | 5.27 | 39.54 | 0.33 | 13.88 / 21.15 | 13.3 |
| `--show`, run 2 | 60.15 / 70.93 | 47.22 / 55.05 | 12.96 | 3.81 | 32.99 | 0.27 | 10.16 / 13.36 | 16.6 |

In run 1 the other steps barely moved against the headless run, so its drop is the `draw` step. Run 2 was faster at every step, `detect` included, which the window can't speed up, so the laptop itself was running faster then (clock speed and temperature vary on a laptop). Compare `draw` across runs, not FPS. The `draw` step covers drawing the HUD on a copy of the 1080p frame, converting it to RGB, uploading it as an OpenGL texture, building and rendering the ImGui frame, and polling the window's events. We haven't split it further, so how much of it is the upload and how much the HUD isn't known. The window swaps buffers without waiting for the monitor's refresh (`glfwSwapInterval(0)`). Waiting would cap the frame rate at the refresh rate and hide the wait inside `draw`.

## Caveats

- One headless run and two window runs, on one laptop. Here the same step varied by up to 30% between runs (`prep` 5.27 vs 3.81 ms), far more than the few percent seen in [#11](cpp-pipeline-speed.md).
- The window was open but not looked at: this session's desktop was locked. Rendering doesn't depend on that, but a visible window on a different display could cost a little more or less.
- The panel shows the frames up to the one before the frame on screen. A frame's own `draw` time isn't known until it's on screen, and the step times reach the panel after that.
- The first frames include warm-up (model load, window creation). They drop out of the panel after 300 frames but stay in the whole-run summary.
