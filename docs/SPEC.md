# Tether — Spec

Real-time multi-object tracker in C++ that keeps stable IDs on fast-moving and temporarily blocked objects.

## Problem Statement

Tracking objects in live video is easy when they move slowly and stay in view. It breaks down in the cases that matter most for defense, drones, and robotics:

- **Fast targets.** An object moves so far between frames that the tracker thinks it is a new object, and its ID changes.
- **Blocked targets.** An object goes behind something, the tracker drops it, and it comes back with a new ID.
- **Speed limits.** Python prototypes are fine for ideas, but the per-frame "glue" logic (matching, prediction, ID management) becomes a bottleneck at high frame rates, and garbage-collection pauses cause dropped frames.

As a student building a portfolio for defense and perception roles, I also need a project that shows the skills these job listings ask for: C++, real-time pipelines, object tracking, ONNX, TensorRT, CUDA, and honest benchmark numbers.

## Solution

Tether is a real-time tracker that:

1. Finds objects automatically in each frame using a pretrained YOLO detector.
2. Follows every object with its own stable ID, using a hand-written tracker that predicts motion so it can survive fast movement and short blockages.
3. Runs as a fast C++ pipeline on an NVIDIA GPU (ONNX model, TensorRT, custom CUDA frame prep, parallel stages).
4. Shows a live heads-up display: boxes, IDs, trails, predicted positions, "searching" markers, and a stats panel with FPS, worst-case frame time, and end-to-end delay.
5. Proves its claims with benchmarks: accuracy on MOT17 against ByteTrack, robustness on Anti-UAV drone footage, and before-vs-after speed numbers for every upgrade.

It is built in two passes: a Python prototype to get the tracking logic right, then a C++ version whose output matches it and runs much faster.

## User Stories

### Live demo

1. As a demo viewer, I want to point a webcam at a scene and see objects boxed automatically, so that I don't have to select anything by hand.
2. As a demo viewer, I want every object to have a visible ID number, so that I can tell objects apart.
3. As a demo viewer, I want an object's ID to stay the same as it moves around, so that I can trust the tracker is following the same thing.
4. As a demo viewer, I want an object to keep its ID after it walks behind something and comes back, so that I can see the tracker handles blocking.
5. As a demo viewer, I want a fast-moving object (like a tossed ball) to keep its ID, so that I can see the tracker handles speed.
6. As a demo viewer, I want to see a short fading trail behind each object, so that I can see where it has been.
7. As a demo viewer, I want to see a dashed box where the tracker predicts an object will be next, so that I can see the tracker "thinking ahead."
8. As a demo viewer, I want a blocked object to show a distinct "searching" marker with how long it has been missing, so that I can tell lost objects from tracked ones.
9. As a demo viewer, I want each object to have its own color, so that neighboring objects never look alike.
10. As a demo viewer, I want status shown by line style (solid, dashed, accent-colored dashed) and not by color alone, so that the display stays readable for everyone.
11. As a demo viewer, I want a large live FPS number, so that I can see the speed claim at a glance.
12. As a demo viewer, I want to see average frame time, worst 1% frame time, and end-to-end delay, so that I understand speed beyond a single average.
13. As a demo viewer, I want a per-step time breakdown (capture, frame prep, detect, track, draw), so that I can see where time is spent.
14. As a demo viewer, I want a list of current tracks with their class and status, so that I can read the scene without hunting on the video.
15. As a demo viewer, I want to see which speed settings are active (TensorRT, CUDA prep, thread count), so that I know what produced the numbers.
16. As a presenter, I want to pause the live view, so that I can explain what is on screen.
17. As a presenter, I want to record the display to a video file, so that I can make a demo video for my portfolio.

### Inputs

18. As a user, I want to run Tether on a live webcam, so that I can demo it anywhere with a laptop.
19. As a user, I want to run Tether on a video file, so that I can test on datasets and replay the exact same clip after every change.
20. As a user, I want video files to be processed as fast as the pipeline allows (not capped at real time), so that I can measure true pipeline speed.
21. As a user, I want the live webcam mode to keep up with the camera's full frame rate without dropping frames, so that the demo stays smooth.

### Tracking behavior

22. As a developer, I want the tracker to take a list of detections per frame and return a list of tracks, so that it can be tested on its own without video or a model.
23. As a developer, I want each track to report its ID, box, class, status, and predicted next box, so that the display and the evaluation tools get everything they need.
24. As a developer, I want new detections to start as tentative tracks until confirmed over a few frames, so that one-frame false detections don't create IDs.
25. As a developer, I want a track that loses its detection to switch to "searching" and keep predicting its position, so that it can be re-matched when the object returns.
26. As a developer, I want a searching track to be removed after a set number of frames, so that dead tracks don't pile up.
27. As a developer, I want matching to compare new detections against each track's predicted position (not its last seen position), so that fast objects still match even when their boxes don't overlap between frames.
28. As a developer, I want the same prediction system to handle both blocking and fast motion, so that there is one motion model to tune and test.
29. As a developer, I want the tracker's behavior to be deterministic for the same input, so that test results and Python-vs-C++ comparisons are repeatable.

### Python prototype and C++ port

30. As a developer, I want to build the tracker in Python first, so that I can get the logic right quickly.
31. As a developer, I want the Python and C++ trackers to share the exact same input and output format, so that they can be compared directly.
32. As a developer, I want a check that the Python and C++ trackers produce matching tracks for the same input (same IDs and statuses every frame, boxes within a small tolerance), so that I know the C++ port is correct.
33. As a developer, I want to time the tracking step alone in both languages, so that I can show the speedup on my own code.
34. As a developer, I want to time the whole pipeline in both languages, so that I can report the overall speedup honestly.

### Detector and runtime

35. As a developer, I want to use a pretrained YOLO26n model exported to ONNX, so that I get good detection with no training.
36. As a developer, I want to run the model with ONNX Runtime first, so that I can get the pipeline working with the simplest setup.
37. As a developer, I want to switch the model runtime to TensorRT, so that it runs as fast as possible on my GPU.
38. As a developer, I want to choose the runtime with a setting, so that I can compare ONNX Runtime and TensorRT on the same video.
39. As a developer, I want to swap model sizes (n, s, m) with a setting, so that I can build a speed-vs-accuracy comparison table.

### Performance work

40. As a developer, I want a simple one-step-at-a-time pipeline first, so that it is easy to debug.
41. As a developer, I want to then split the pipeline into parallel stages (capture, detect, track, draw) connected by bounded queues, so that stages overlap and FPS goes up.
42. As a developer, I want to switch between sequential and parallel modes with a setting, so that I can measure the difference.
43. As a developer, I want to profile the pipeline with NVIDIA's profiler before writing GPU code, so that I can prove frame prep is a real bottleneck.
44. As a developer, I want a custom CUDA kernel for frame prep (resize, color conversion, scaling), so that frames are prepared on the GPU and stay there for the model.
45. As a developer, I want to switch between CPU and CUDA frame prep with a setting, so that I can measure the speedup and keep a CPU path for non-NVIDIA devices.
46. As a developer, I want a check that CUDA and CPU frame prep give the same result within tiny rounding differences, so that I know the kernel is correct.

### Measurement and evaluation

47. As a developer, I want every run to log per-step timing for every frame, so that I can compute average, worst 1%, and end-to-end delay.
48. As a developer, I want the tracker to write its output in the standard MOT results format, so that MOT17 can be scored with standard tools.
49. As a developer, I want to score my tracker on MOT17 with HOTA, IDF1, and ID switches, so that my results are comparable to published trackers.
50. As a developer, I want to run ByteTrack on the exact same detections, so that the comparison measures tracking logic and not the detector. The detections are saved unfiltered (down to a very low confidence, e.g. 0.1), and each tracker applies its own confidence thresholds internally, so ByteTrack keeps the low-confidence second matching pass it depends on.
51. As a developer, I want a tool that feeds Anti-UAV's labeled drone boxes into the tracker with some boxes randomly deleted and some stretches blanked out, so that I can stress-test fast-motion and blocking recovery without a drone detector.
52. As a developer, I want to control the deletion rate and gap length in that tool, so that I can test specific failure conditions.
53. As a developer, I want to measure prediction error during gaps, so that I can show the motion model improving over time.
54. As a developer, I want benchmark results saved as files, so that I can build README tables from them.

### Build and setup

55. As a developer, I want the project to build with CMake on Windows, so that I can develop and demo on my laptop.
56. As a developer, I want a Docker container (Linux + GPU) that builds and runs the benchmarks on video files, so that the project runs on Linux and others can reproduce my numbers with one command.
57. As a recruiter or reviewer, I want a README with the demo video, the benchmark tables, and how to run it, so that I can understand the project in a few minutes.
58. As an open-source user, I want a clear license, so that I know how I can use the code.

### Drone detector (stage 5)

59. As a developer, I want a val split made of whole recording sessions, so that the detector is never tuned on near-copies of its own training footage.
60. As a developer, I want to score any detector on Anti-UAV clips (AP50, recall, precision and F1 at a threshold, recall by box size, false alarms per minute on drone-absent frames), so that I can compare the pretrained model and the fine-tuned one on the same footing.
61. As a developer, I want a single-class drone model fine-tuned on Anti-UAV, used by any tool through `--model`, so that Tether finds drones itself while the COCO model and every MOT17 result stay as they are.
62. As a developer, I want the tracker scored on the drone model's own detections, next to the same clips' labeled boxes, so that I can see what a real detector costs.
63. As a developer, I want the detector scored on a dataset it never trained on, and on images with birds and no drones, so that I know whether it generalizes and how often it calls something else a drone.
64. As a viewer, I want the HUD and stats panel to show the drone model's class name, so that a drone isn't labeled "person".

## Implementation Decisions

### Stages

- **Stage 1: Python prototype.** Detection, tracker, IDs, blocking and fast-motion handling, basic display.
- **Stage 2: C++ version.** Same tracker logic, matching outputs, sequential pipeline first, then parallel stages.
- **Stage 3: Benchmarks.** MOT17 vs. ByteTrack, Anti-UAV dropout tests, Python vs. C++ timing.
- **Stage 4: GPU speed.** Profiling, TensorRT, custom CUDA frame prep.
- **Stage 5: Drone detector.** A single-class YOLO26n drone model fine-tuned on Anti-UAV and scored end to end, on a session-disjoint val, with a transfer test and false-alarm rates. See `docs/adr/0004-drone-detector-on-anti-uav.md` (milestone "Drone detection", #41–#48).
- Stages 1–4 are "done for the resume." Stage 5 builds on them. Stage 6 (Android) is a later upgrade.

### Modules

- **Video source.** Reads frames from a webcam or a video file through one interface. File mode runs uncapped; webcam mode runs at the camera's rate.
- **Frame prep.** Turns a raw frame into the model's input (resize, color order, scale). Two versions behind one interface: CPU and CUDA. The CUDA version keeps the result on the GPU for the detector.
- **Detector.** Loads an ONNX model and returns detections (box, score, class) per frame. Two runtimes behind one interface: ONNX Runtime and TensorRT. Starts with YOLO26n; model size is a setting.
- **Tracker (core module).** Takes a frame's detections, returns the current tracks. Hand-written. Contains:
  - **Motion model:** a Kalman filter per track that predicts position and velocity.
  - **Matching:** compares detections to each track's *predicted* box and finds the best overall pairing (Hungarian algorithm).
  - **Track lifecycle:** tentative → tracked → searching → removed, with set thresholds for confirming and dropping tracks.
  - Deterministic: the same input always gives the same output.
  - All limits are counted in frames, never wall-clock time, so results don't change with pipeline speed. Where a limit is stated in seconds, it is converted using the video's own frame rate.
- **Pipeline.** Connects the modules. Two modes: sequential, and parallel stages with bounded queues between them. Logs per-step timing for every frame.
- **Display (HUD).** Draws boxes, IDs, trails, predicted boxes, and searching markers with OpenCV, and the stats panel with Dear ImGui. Supports pause and record.
- **Evaluation tools.** Write MOT-format results; run MOT17 scoring; run the Anti-UAV dropout simulator; compute timing summaries (average, worst 1%, end-to-end delay).
- **Python prototype.** Same tracker interface and output format as the C++ tracker.

### Tracker interface

- **Detection:** box, confidence score, class.
- **Track:** stable ID, current box, class, status (tentative, tracked, searching), predicted next box, frames since last seen, recent path for trails.
- Input and output can be saved to and loaded from files, so the tracker can be run and compared without video or a model.

### Technology

- **Detector model:** Ultralytics YOLO26n at 1280×736 input (no NMS step, built for edge devices, better on small objects). n, s and m were compared at 640 and 1280×736 in #18; see `docs/adr/0003-ship-yolo26n-1280x736.md`.
- **Model format:** ONNX.
- **Runtimes:** ONNX Runtime first, then TensorRT on the laptop GPU. ONNX Runtime is also the plan for Android later.
- **GPU code:** CUDA, for frame prep only. The tracker stays on the CPU, since it is too small a job for the GPU to help.
- **Display:** OpenCV for drawing on video, Dear ImGui for the stats panel.
- **Build:** CMake, with code kept portable between Windows and Linux.
- **Platforms:** Windows native for development and live webcam demos. Docker (Linux + NVIDIA GPU, built from NVIDIA's CUDA/TensorRT images) for benchmarks on video files.
- **License:** AGPL-3.0, to match the Ultralytics model license.

### Visual design

- Near-black background, off-white text, one accent color (signal orange) used for the "searching" state, CUDA highlights, and live indicators.
- IBM Plex Mono for numbers and labels, IBM Plex Sans for regular text.
- Corner-bracket boxes for tracked objects, dashed boxes for predictions, accent-colored dashed boxes for searching objects.
- Each track gets its own color, spread evenly around the color wheel.
- Stats panel: big FPS number, then average frame time, worst 1% frame time, end-to-end delay, per-step bars, and the track list.

## Testing Decisions

### What makes a good test

Tests check what the code does from the outside, not how it does it internally. A test gives a module an input and checks its output. Internal details (like the Kalman filter's exact numbers mid-step) are not tested directly, so the internals can change freely as long as behavior stays correct.

### Seams

1. **Tracker (main seam).** Detections in, tracks out. Covers:
   - **Behavior tests** with scripted detection sequences: an object disappears for N frames and returns (same ID expected); an object jumps far every frame (same ID expected), including one that is already moving fast on its first frame, before the motion model has any velocity history; two objects cross paths (IDs must not swap); a one-frame false detection (no confirmed track expected); a long disappearance (track removed after the limit).
   - **Python vs. C++ match:** the same input file through both trackers must give the same IDs and statuses on every frame, with boxes equal within a small tolerance (floating-point math differs slightly between languages).
   - **MOT17 and Anti-UAV scoring**, and the **ByteTrack comparison**, all run through this same seam.
2. **Frame prep (small seam).** Raw frame in, prepared model input out. The CUDA output must match the CPU output within small rounding differences.

**Speed** is measured, not pass/fail tested: run the whole app on video files and log per-step timing.

There is no existing codebase, so there is no prior art. The scripted tracker tests become the pattern all later tests follow.

### Targets

All speed targets are for the final C++ version running the shipped configuration: YOLO26n at 1280×736 input on TensorRT, with CUDA frame prep and the parallel pipeline (`tether-pipeline VIDEO --runtime tensorrt --prep cuda --pipeline parallel`), on an RTX 5070 laptop GPU. They were first written for YOLO26n at 640; ADR 0003 moved them to the model that ships. All targets are starting goals. Real numbers are reported either way.

**Speed**
- Pipeline speed on video files: 120+ FPS.
- Worst 1% frame time (P99): under 16 ms.
- End-to-end delay on webcam: under 50 ms.
- Live webcam: keep up with the camera's full rate with zero dropped frames.

**Accuracy on MOT17** (same unfiltered detections given to both trackers; each applies its own thresholds). The headline detection set is YOLO26n at 1280×736 input, which is also the shipped detector, with YOLO26n at 640 (the shipped detector until ADR 0003) reported as a second row. See `docs/adr/0002-mot17-benchmark-detections-yolo26n-1280.md` and `docs/adr/0003-ship-yolo26n-1280x736.md`.
- HOTA: within 2 points of ByteTrack.
- IDF1: equal to or better than ByteTrack.
- ID switches: equal to or fewer than ByteTrack.

**Anti-UAV dropout tests** (labeled boxes with failures added on purpose)
- An ID only counts as kept if the track's box overlaps the labeled drone box with IoU ≥ 0.5. Otherwise a track that never dies would score 100% on single-drone clips.
- With 20% of detections randomly deleted: same ID kept on at least 95% of the frames a perfect tracker could keep. That is, ID kept ≥ 0.95 × the reachable ceiling, computed on the same deleted boxes with the tracker's own search window (`tools/antiuav_ceiling.py`). Report the ceiling next to the score every time.
  - Why relative: a track forgotten after its search window must take a new ID when the drone leaves the labels for longer, so a flat 95% sat above what any 30-frame tracker can reach on test (94.0%). Lengthening the window instead was rejected: on test only 3 s clears 95%, and a longer search gives dead tracks more time to be revived on the wrong object in crowds. See `docs/benchmarks/antiuav-ceiling.md` (#33).
  - The ceiling ignores the frames a new track needs to be confirmed, so even a perfect tracker scores slightly under 100% of it.
- With gaps up to 1 second (converted to frames using the clip's frame rate): same ID recovered at least 80% of the time.
- Prediction error during gaps: reported as a result, not pass/fail.

**Drone detector** (stage 5, ADR 0004)
- Speed: the drone model in the shipped configuration meets the speed targets above (120+ FPS, P99 under 16 ms) on an Anti-UAV clip. Met in #46: 195.59–198.63 FPS and P99 6.46–6.49 ms on the first val clip, since the earlier speed clip is in test (`docs/benchmarks/drone-speed.md`).
- Every number is reported against two references: the pretrained COCO YOLO26n scored class-agnostically (the "before" row), and the labeled-box tracking results (a reference, not a strict ceiling; see End-to-end tracking below).
- Scoring: a detection is a hit at IoU ≥ 0.5 with the labeled box. Models are compared on AP50. Recall, precision and F1 are reported at the threshold with the best F1 on the session-disjoint val, frozen before test.
- End-to-end tracking: the tracker runs on the detector's own detections (class-agnostic, nothing deleted on top) and is scored with the dropout tests' ID-kept rule on every frame where the drone is labeled, so the detector's misses count against it. The same clips' labeled boxes are run beside it. The tracker's cutoff is the detector's operating threshold, and its low-score band's floor is 0.4 × the cutoff (0.1 at the default 0.25). See `docs/benchmarks/drone-tracking.md` (#44).

**Drone detector accuracy targets** (set from val before test, #47; full reasoning in ADR 0004, "Accuracy targets")

The drone model `yolo26n_drone_1280x736` at the frozen threshold 0.352, TensorRT, scores saved down to 0.01, scored once on test in #48. Val chose the epoch, the threshold and the tracker's cutoff, and it's one site and one camera. Even on labeled boxes, the tracker scores 87.0% on val and 82.6% on test. So no target is set just under its val score. Each one rests on something else, stated with it.
- **Detector recall on Anti-UAV test: ≥ 0.80** (pass/fail).
  - Why: the dropout test's 20% deletion is the miss rate the tracker is specified to cope with, so a detector at ≥ 0.80 keeps it inside that regime.
  - Val: 0.992 (`docs/benchmarks/drone-model.md`). COCO: 0.142 at 0.25 (`docs/benchmarks/drone-coco-baseline.md`).
  - Passing doesn't show the misses are spread out like random deletions. The ID-kept target catches runs of misses.
- **Detector precision on Anti-UAV test: ≥ 0.90** (pass/fail).
  - Why: that's at most about 0.09–0.11 false boxes per frame, and a false box needs 3 hits in a row to become a track.
  - Val: 0.995 (`drone-model.md`). COCO: 0.410 at 0.25.
  - Passing says nothing about empty sky or birds.
- **AP50 on Anti-UAV test: reported, no separate target.** The two targets above already imply AP50 ≥ 0.72 (0.80 × 0.90 under the precision envelope). Val: 0.995, against COCO's 0.228 (`drone-model.md`).
- **End-to-end ID kept on Anti-UAV test: ≥ 0.95 × the labeled-box row on the same clips, in the same run** (pass/fail). The clips where the detector beats the labels are reported beside it, along with the reachable ceiling.
  - Why: it's the dropout test's relative rule applied to a real detector. It's relative because test is harder on labels alone.
  - The labeled-box row is a reference, not a ceiling. On val the drone model beat it on 4 clips where the labels lose the drone, so passing doesn't prove the detector is near the best possible.
  - Val: 89.7%, against 87.0% on labels (`docs/benchmarks/drone-tracking.md`).
- **Bird false-alarm share: ≤ 8 of the 86 Drone-vs-Bird bird images (≤ 10%)** (pass/fail, coarse). A coarse bar that only catches a gross failure.
  - Resolution: one image is about 1.2 points, the noise is about ±6 points, and the images cluster in a few videos.
  - Val: none. COCO: 96.5% at 0.25 (`docs/benchmarks/drone-transfer-sets.md`).
  - Passing doesn't show the model can tell a distant bird from a distant drone.
- **Empty-sky false alarms on Anti-UAV test: ≤ 4 detections in its 2.25 minutes (≤ 2 per minute)** (pass/fail, coarse).
  - Resolution: one alarm is about 0.44 per minute, counted as detections, not tracks, so the bar of 4 is 1.8 per minute. 4 lets one short false object pass.
  - Val: 0 in 0.50 min, partly by construction, since the 2 s rule was sized on val's furthest drone-absent detection (36 frames against the 40-frame margin) (`drone-transfer-sets.md`). COCO at 0.066: 17.9 per minute.
  - Passing covers one site's empty sky only.
- **Transfer to Seraphim test: AP50 above the COCO model's 0.216 on the same 8,349 images, runtime and floor** (pass/fail). Recall and precision at 0.352 are reported beside it.
  - Why it's relative to COCO: Seraphim mixes real, marketing and synthetic images, padded to 640×640, so an absolute bar has nothing to stand on. Beating the model it was fine-tuned from is the minimum for "it learned drones, not the site". A failure makes training on Seraphim the next experiment.
  - Val: none (`drone-transfer-sets.md`).
- **Recall on tiny and small drones: reported, no target.** Val has 8 tiny and 12 small boxes. The drone model finds 0 of 8 and 3 of 12 (`drone-model.md`), too few to judge.

**Result on test (#48, run once; `docs/benchmarks/drone-test.md`):**
- recall 0.884: **met**;
- precision 0.975: **met**;
- AP50 0.915 (COCO 0.402);
- ID kept 75.3% against 82.6% on the labels, 0.911 ×: **missed**;
- birds 19 of 86: **missed**;
- empty sky 190 detections in 2.25 min: **missed**;
- Seraphim AP50 0.164 against COCO's 0.216: **missed**;
- tiny 0 of 39, small 397 of 823.

**Upgrade payoffs**
- Tracking step, C++ vs. Python: 10x+ faster.
- Whole pipeline, C++ vs. Python: 2x+ faster.
- Parallel vs. sequential pipeline: 1.5x+ faster.
- CUDA frame prep vs. CPU frame prep: 3x+ faster on that step, plus total time saved per frame.
- TensorRT vs. ONNX Runtime: 1.5x+ faster model run.
- Python and C++ trackers give matching outputs on the same input (as defined in Seams).

## Out of Scope

- **Android app** (stage 6, later). The C++ core and ONNX model are chosen so this port is straightforward.
- **NVIDIA Jetson deployment** (possible later upgrade, a candidate for a later drone milestone).
- **Training or fine-tuning any model** in stages 1–4. Stage 5 fine-tunes only the drone detector (ADR 0004).
- **Training the drone detector on anything but Anti-UAV.** Seraphim is held out as the transfer test. Training on it is the next experiment if the transfer score is poor.
- **Bearing output and a low-latency mode** (candidates for a later drone milestone).
- **Running the tracker itself on the GPU.**
- **Custom GPU duplicate-box removal (NMS)**, not needed since YOLO26 has no NMS step.
- **Re-identifying the same object across different cameras.**
- **Thermal or infrared video** (a candidate for a later drone milestone; the local Anti-UAV300 copy is visible light only).
- **Click-to-select single-target tracking.** Detection is automatic.
- **Live webcam inside Docker.** Docker is for video-file benchmarks only.

## Further Notes

- **Webcam speed is capped by the camera** (usually 30 or 60 FPS). True pipeline speed is always measured on video files.
- **Most drone tracking videos show one drone at a time.** That is why MOT17 covers multi-object accuracy and Anti-UAV covers fast, small targets.
- **The detector dominates frame time**, so whole-pipeline gains from C++ will be smaller than tracking-step gains. Both numbers are reported.
- **Small, far, fast targets** pull the model-size choice in opposite directions (faster model vs. more accurate model). The n/s/m comparison is meant to show that trade-off.
- **Input size matters more than model size for small targets.** On MOT17, going from 640 to 1280 input more than doubled detector recall, while bigger models at 640 barely helped on the two videos first checked. The n/s/m comparison therefore also varied input size. On all 7 videos (#18), bigger models help too (s at 640: 40.0% recall against n's 33.7%), but less than input size (n at 1280×736: 54.6%).
- **License:** Ultralytics models use AGPL-3.0, which is fine for an open-source project.
- **Name:** Tether.
