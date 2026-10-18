# Concepts

A running list of the ideas behind Tether, in plain English. Each entry says what the thing is, then how it shows up in this project. Add to it as new ideas come up.

(This is a learning notebook. The canonical project vocabulary for agents lives in `CONTEXT.md`, if and when it exists.)

---

## Tracking basics

### Multi-object tracking (MOT)
Following many objects through a video and giving each one a stable ID from frame to frame. The hard part isn't finding objects; it's deciding which object in this frame is *the same one* as an object in the last frame.

**In Tether:** the whole project. "Stable IDs" is the product.

### Tracking-by-detection
The most common way to do MOT. A detector finds objects in each frame independently and knows nothing about IDs. A separate tracker then links those detections across frames. The two jobs are separate, so you can swap either one.

**In Tether:** YOLO is the detector, and the hand-written tracker does the linking. That split is why the tracker can be tested with no video or model at all: you just feed it detection files.

### Detection vs. track
- A **detection** is one box in one frame: position, confidence score, class. It has no memory.
- A **track** is an object's identity over time: an ID, a current box, a predicted next box, a status, and a history.

The tracker's job each frame is to pair detections with tracks.

### Occlusion
An object being hidden behind something else, in whole or in part. The detector loses it, and the tracker has to keep the track alive and guess where the object went until it comes back.

**In Tether:** the spec calls these "blocked targets." Tracks in this situation are in the *searching* state.

### ID switch
When the tracker gives an object a new ID, or swaps IDs between two objects. It's the main failure tracking tries to avoid, and it's counted directly as a metric.

### Track lifecycle
The states a track moves through:
- **Tentative:** just appeared. It isn't trusted yet because it could be a one-frame false detection.
- **Tracked:** confirmed after being matched for a few frames in a row.
- **Searching:** lost its detection. It keeps predicting where the object is and waits to be re-matched.
- **Removed:** searched too long without a match, so it's deleted.

**In Tether:** every threshold is counted in **frames**, not seconds (see *Frame-based vs. wall-clock limits*). The defaults are 3 frames to confirm and 30 missed frames before removal (`--confirm-frames`, `--max-missing-frames`). A tentative track that misses even one frame is dropped on the spot. A searching track reports its last *seen* box as `box` and its guess as `predicted_box`. All states compete in one matching pass.

**Gotcha:** "30 frames" is 1 second at 30 FPS but only half a second at 60 FPS. The Anti-UAV goal of recovering 1-second gaps needs the limit set from each clip's frame rate, not left at the default.

---

## Matching & motion

### IoU (Intersection over Union)
A score from 0 to 1 for how much two boxes overlap: the area they share divided by the area they cover together. 1 means identical boxes, and 0 means they don't touch.

**Gotcha:** IoU is 0 for *any* pair of boxes that don't overlap, whether they're 5 pixels apart or 500. Once boxes stop overlapping, it can't tell "close" from "far." That's why fast objects are hard for IoU-based matching.

**In Tether:** the matching cost is roughly `1 − IoU` between a detection and a track's predicted box.

### Hungarian algorithm (optimal assignment)
Given a grid of costs (every track against every detection), it finds the one-to-one pairing with the lowest **total** cost. The alternative is greedy matching, where you take the single best pair, then the next best, and so on. That can make one locally good choice that forces a bad one elsewhere. Hungarian looks at the whole grid at once.

Pairs whose cost is too high (for example IoU below a minimum) are rejected after assignment. Without that rule, the algorithm would happily pair things that are nowhere near each other just because every detection "has to" go somewhere.

### Kalman filter (motion model)
A small piece of math that keeps a running estimate of an object's state (position, size, velocity) plus *how uncertain* that estimate is. Each frame it does two steps:
1. **Predict:** move the estimate forward using the velocity ("it was here and moving right, so now it's probably there"). Uncertainty grows.
2. **Update:** when a matching detection arrives, blend the prediction with the measurement. How much it trusts each side depends on their uncertainties, so a noisy detection gets less weight. Uncertainty shrinks.

When no detection arrives (occlusion), it keeps predicting with no updates. That's how a searching track "follows" an object it can't see.

**Gotcha:** a brand-new track has no velocity history, so its first predictions assume the object is standing still. That's the weak spot for objects that are already fast when they appear.

### Predicted-box matching
Matching each new detection against where a track is **predicted** to be now, not where it was last seen. For a fast object, the last-seen box may not overlap the new detection at all (IoU = 0), but the predicted box will if the motion model is doing its job.

**In Tether:** one motion model handles both fast motion and occlusion. In both cases the answer is "trust the prediction."

### Process noise (how fast the filter adapts)
The Kalman filter's **process noise** says how much the true motion may change between frames that the constant-velocity model can't explain. Low process noise means "trust the model": smooth, but slow to notice the object speeding up. High process noise means "trust the detections": it reacts fast, but it's jumpier.

**Gotcha:** trackers built for walking pedestrians use very low velocity noise. Applied to something that speeds up quickly, the prediction lags behind the object, the predicted box stops overlapping the detection, and the ID is lost.

**In Tether:** center velocity noise is `1/10` of the box height per frame. `1/160`, the usual pedestrian value, failed the speed-up test. Size velocity keeps `1/160`: real boxes change size slowly, and with high size noise a single smaller (partial) detection was read as a fast shrink, which predicted an inside-out box. See `docs/adr/0001-center-distance-fallback-for-new-tracks.md`.

### Center-distance fallback
When boxes don't overlap, IoU is 0 whether they're near or far, so it can't pick between them. Center distance can. In Tether it is used only for tracks seen once, which have no velocity yet: a second matching pass by the distance between box centers, measured in box sizes and capped.

---

## Evaluation

### MOT17
A standard pedestrian-tracking benchmark from MOTChallenge: crowded street videos with hand-labeled boxes and IDs. Scoring on it makes results comparable to published trackers.

### Anti-UAV
A dataset of drone videos with labeled drone boxes. Drones are small, fast, and often in single-target clips, so it tests different things than crowded MOT17 does.

**In Tether:** we don't run a detector on it. We feed in the *labeled* boxes, delete some on purpose, and see whether the tracker recovers.

### HOTA (Higher Order Tracking Accuracy)
A tracking metric that balances two things: how well you **detect** objects, and how well you **associate** them over time (keeping IDs right). It's the geometric mean of the two, so you can't score well by being great at one and bad at the other.

### IDF1
An F1 score for identity: of all the boxes, how many carry the *right* ID over the long run. It's very sensitive to ID switches and track fragmentation.

### ByteTrack
A well-known, simple, strong tracker, used as the baseline. Its key idea is that low-confidence detections, which most trackers throw away, still get a **second matching pass** against tracks left unmatched by the first pass. That's how it holds onto half-hidden people.

**In Tether:** both trackers get the same *unfiltered* detections, so ByteTrack keeps its second pass and the comparison stays fair.

### Null-case test
Before trusting a metric, ask: *what score would a useless system get?* If a tracker that does nothing smart still scores well, the metric isn't measuring what you think.

**In Tether:** a tracker that never drops its one track would score 100% "same ID kept" on single-drone clips. The fix was to require IoU ≥ 0.5 with the real drone box for the ID to count.

---

## Detection & runtimes

### YOLO (and YOLO26)
A family of fast single-pass object detectors. YOLO26 is Ultralytics' version that is **NMS-free**: it doesn't need a separate cleanup step to remove duplicate boxes, which simplifies the pipeline and removes a step that can be slow.

### NMS (Non-Maximum Suppression)
The traditional cleanup step after detection. When several boxes cover the same object, it keeps the most confident one and drops the rest. YOLO26 doesn't need it.

### Confidence threshold
The minimum detector confidence a detection must have to be used. A high threshold means fewer false boxes but more missed objects; a low one means the opposite. *Where* you apply it matters (see ByteTrack).

### ONNX
A standard file format for trained neural networks. You export the model once and many runtimes can load it, which is what makes the runtime swappable.

### ONNX Runtime vs. TensorRT
Both run an ONNX model.
- **ONNX Runtime** is general-purpose, easy to set up, and runs almost anywhere (including Android later).
- **TensorRT** is NVIDIA-only. It compiles the model specifically for *your* GPU (fusing layers, picking the fastest kernels, using lower precision), so it's usually much faster but less portable.

### CUDA kernel
A function that runs on the GPU across thousands of threads at once. It's a natural fit for per-pixel work like resizing and color conversion.

### Frame prep (preprocessing)
Turning a raw camera frame into what the model expects: resize to the input size, reorder color channels (OpenCV uses BGR, models usually want RGB), and scale pixel values (e.g. 0–255 down to 0–1).

### Host vs. device memory
"Host" means CPU RAM and "device" means GPU memory. Copying between them costs time. Doing frame prep on the GPU and handing the result *straight* to the detector skips a round trip.

---

## Performance

### Throughput vs. latency
- **Throughput:** how many frames per second come out (FPS).
- **Latency:** how long one frame takes from camera to screen (end-to-end delay).

They're different. A parallel pipeline can raise FPS without making any single frame faster, and can even add a little delay per frame.

**In Tether:** the timing log stores each step's start and end time, not just how long it took, so both can be measured from one log. Frame time is the gap between frames finishing; delay is from a frame being captured to it being drawn. In the sequential pipeline they're nearly the same number. In the parallel one they split apart, and only timestamps can tell.

**Gotcha:** the measured delay starts when the program has the frame. The camera's own sensor-to-USB delay comes before that and isn't counted, so a true camera-to-screen number is higher.

### Tail latency (P99 / worst 1%)
The frame time that 99% of frames beat. Averages hide the occasional slow frame, but the viewer sees it as a stutter. A pipeline averaging 8 ms with a 40 ms P99 *feels* worse than a steady 12 ms.

### Nearest-rank percentile
One way to compute a percentile: sort the values and pick the one at position ⌈p × n⌉. The answer is always a value you actually measured. NumPy's default instead *interpolates* between neighbors, which can give a frame time no frame ever had. On small samples the two disagree a lot.

**In Tether:** `tether-timing` uses nearest rank. With under 100 frames, P99 is simply the worst frame.

### Sequential vs. parallel (pipelined) stages
- **Sequential:** capture → prep → detect → track → draw, one frame at a time. Simple to debug, but every stage waits on the others.
- **Parallel:** each stage runs on its own thread and works on a different frame at the same moment, like an assembly line. Throughput becomes limited by the *slowest single stage*, not the sum of all of them.

### Bounded queue & backpressure
The buffers between pipeline stages have a fixed maximum size. If a slow stage falls behind, its input queue fills up and the faster stage before it has to wait. That pushback is **backpressure**. Without a limit, a fast producer would pile up frames forever, using more and more memory and adding more and more delay.

### Amdahl's law
Speeding up one part of a job only helps in proportion to how much of the total time that part takes. If tracking is 5% of frame time, making it 10× faster saves at most about 4.5%.

**In Tether:** this is why the spec reports the tracking-step speedup (expected to be large) and the whole-pipeline speedup (expected to be smaller, because the detector dominates) as separate numbers.

### Profiling (Nsight Systems)
Measuring where time actually goes before optimizing. Nsight Systems is NVIDIA's timeline profiler: it shows CPU threads, GPU kernels, and memory copies side by side.

**In Tether:** the rule is to profile *before* writing the CUDA kernel, to prove frame prep is a real bottleneck instead of assuming it.

---

## Python vs. C++

### Why C++ is faster for the "glue"
- **Compiled vs. interpreted:** C++ becomes machine code ahead of time. Python's interpreter does extra work on every line: looking up types, checking them, and dispatching. Tight per-frame loops pay that cost thousands of times.
- **Memory layout:** a C++ array of structs sits in one contiguous block, which keeps the CPU cache happy. A Python list holds pointers to objects scattered around memory.
- **Predictable timing:** C++ frees memory exactly when you say so. Python mostly frees memory right away (reference counting), but its cycle collector occasionally runs and pauses the program. Those pauses hurt tail latency.
- **Real parallel threads:** the GIL (Global Interpreter Lock) means standard CPython runs only one thread of Python code at a time, so CPU-heavy stages can't truly run in parallel. Newer "free-threaded" Python builds are removing this, but they're not the default yet.

### Where Python is already fast
Libraries like NumPy and ONNX Runtime do their heavy work in compiled C/C++ under the hood. The slow part of Python is the *glue between those calls*, which is exactly the per-frame matching and bookkeeping that Tether ports to C++.

### Floating-point differences between languages
Decimal math on computers is rounded, and the result can depend on the order of operations. Python (NumPy) and C++ can do the same math in a slightly different order and get answers that differ in the last few digits. Neither one is wrong.

**In Tether:** that's why the Python-vs-C++ check requires *matching* outputs (same IDs and statuses, boxes within a tiny tolerance), not bit-identical ones.

---

## Design & testing ideas

### Determinism
The same input always produces the same output. That's what makes tests and Python-vs-C++ comparisons repeatable. Common ways to lose it include random numbers, iterating in hash order, and anything that depends on the clock.

### Frame-based vs. wall-clock limits
Rules like "drop a lost track after N" should count **frames**, not seconds. On video files the pipeline runs as fast as it can, so a 1-second limit would mean 30 frames in Python but 200 in C++, and the results would change with code speed. If you want seconds, convert them using the video's own frame rate.

### Seam
A boundary where you can feed a module input and check its output without the rest of the system. A good seam makes testing easy.

**In Tether:** the tracker is the main seam (detection files in, track files out). Frame prep is a smaller one (raw frame in, model input out).

### Tracer bullet / vertical slice
Building a thin piece of functionality that works end to end, all the way through every layer, instead of finishing one layer at a time. You get something working early, and integration problems show up early.
