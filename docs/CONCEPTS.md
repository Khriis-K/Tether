# Concepts

A running list of the ideas behind Tether, in plain English. Each entry says what the thing is, then how it shows up in this project. Add to it as new ideas come up.

(This is a learning notebook. The canonical project vocabulary for agents lives in `CONTEXT.md`, if and when it exists.)

The stories behind these ideas (the problems we hit, how we found the causes, and what we learned) are in [LESSONS.md](LESSONS.md).

**Contents:** [Tracking basics](#tracking-basics) · [Matching & motion](#matching--motion) · [Evaluation](#evaluation) · [Detection & runtimes](#detection--runtimes) · [Performance](#performance) · [Python vs. C++](#python-vs-c) · [Design & testing ideas](#design--testing-ideas)

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

### Matching cascade (confirmed tracks first)
Instead of matching every track in one Hungarian pass, match in stages, most trusted tracks first. Each stage only sees the detections the stages before it left over. It's like boarding a plane by group: the passengers with a seat already booked board before standby.

Why bother when Hungarian is "optimal"? It's optimal for the cost you give it, and IoU doesn't know which track deserves a detection more. A tentative track on a duplicate box can overlap a detection just as well as the confirmed track it sits on. In a single pass it can win that detection, and the confirmed person loses their ID. The order you match in is how you tell the matcher which tracks to trust.

**In Tether:** five stages, in this order:
1. Confirmed tracks, by IoU, against detections scored ≥ 0.25.
2. Tracks that were tracked last frame, by IoU ≥ 0.5, against detections scored 0.1–0.25 (the ByteTrack low-score pass).
3. Searching tracks, by the growing search gate.
4. Tentative tracks, by IoU.
5. Tracks seen once, by center distance.

Moving confirmed tracks ahead of tentative ones took MOT17 ID switches from 1270 to 779 (the story is in [LESSONS.md](LESSONS.md#tether-made-470-more-id-switches-than-bytetrack-22-23)).

**Gotcha:** an order that protects one kind of track can starve another. The search gate (stage 3) runs before tentative tracks on purpose: when the drone jumps, its detection starts a tentative track, and the gate has to take that detection back for the searching track that owns it.

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

### Growing search gate
A searching track's prediction drifts further from its object every frame it coasts, so a fixed match threshold eventually can't reach it. A growing gate widens the allowed center distance with each missed frame (`r0 + k·frames_since_seen`, in box sizes).

**In Tether:** `--search-gate` / `--search-gate-growth`, 1.5 + 0.25 by default, with the nearest-track cap below. Uncapped, it raises Anti-UAV 1 s gap recovery from 31% to 88%, but on MOT17 every setting that helps costs HOTA and ID switches.

**Gotcha:** the gate is a recall-vs-precision trade, and the cost only shows up where there are other objects to grab. With one target per clip, a wide gate can't pick the wrong box, so Anti-UAV alone will always say "wider is better". A crowded benchmark is the real test.

### Nearest-track cap
A fixed gate can't be right for both a lone object and a crowd. Capping it at a fraction of the distance to the nearest other track adapts it to the scene: alone, the full gate applies; in a crowd, it shrinks to the space no other track is closer to. At a fraction of one half, two tracks' gates can't overlap.

**In Tether:** `--search-gate-cap`, 0.5 by default, measured to other *confirmed* tracks only. It takes the gate's MOT17 cost from about 500 extra ID switches to none, and costs Anti-UAV about 1 point of ID kept.

**Gotcha:** "alone" means alone among the tracker's tracks, not alone in the world. On Anti-UAV there is one drone, yet the cap bound in about 30% of gate checks, because the tracker's own duplicate IDs of the drone sat nearby. Merging the duplicates (below) took that to 0.6%.

### Duplicate-track merge
When a lost object reappears too far from its track's prediction for the gate, its detection starts a new track, and three frames later the object holds two IDs. A merge folds the new track back into the old one at the moment it is confirmed, if the old one is searching, the same size, and close enough. The older ID survives.

**In Tether:** `--merge-reach` / `--merge-reach-growth`, 10 + 1 box sizes per missed frame by default, capped like the gate. On Anti-UAV val it lifts ID kept at 20% deletion from 59.9% to about 85%, and MOT17 doesn't move: in a crowd the cap holds the reach down to a fraction of the distance to the nearest neighbor.

**Gotcha:** the reach is far wider than the gate, because the drone often reappears 5–10 box sizes from its prediction. That is safe in a crowd (the cap) and on a single-target benchmark (nothing to get wrong), but not in between. In a sparse scene, a lone person who walks out can hand their ID to a same-size newcomer who walks in elsewhere. A scripted webcam scene showed the search gate already does this for gaps of 5–29 frames, because a box-size gate around a big, close-up box spans most of the frame; the merge adds the case where the newcomer arrives on the very next frame.

**Gotcha:** center distance ignores size. A big box walking in front of a smaller, occluded track can have its center inside that track's gate, which IoU would never allow. Tether's gate also requires the heights to be within 1.5 times of each other, against the track's last *observed* box: a coasted prediction's size drifts with the size velocity (MOT17-09 had predictions 1211 px tall, and one with negative height).

### Velocity damping while coasting
A constant-velocity filter coasts a lost track at its last velocity, which overshoots whenever the object turns. Damping multiplies the velocity by a factor below 1 each missed frame, so the prediction slows to a stop instead of running off. 0 is the constant-position null: the box is frozen where it was last predicted.

**In Tether:** `--coast-damping`, off by default (1.0). On Anti-UAV val, constant position ties the best damping on 1 s prediction error, so past about half a second the velocity estimate is useless there. Paired with the search gate, damping lifts 1 s recovery from 88% to about 94%. On MOT17 it is neutral alone and makes the gate worse; with the nearest-track cap it adds ID switches at 640, mostly on moving-camera videos, so it stays off.

**Gotcha:** a lower mean prediction error doesn't mean more re-matches. Without a gate, re-matching needs IoU, which is all-or-nothing within about a box size, and on Anti-UAV without the gate, undamped predictions recovered more gaps even though their average miss was bigger (our unmeasured guess: they land within a box size more often). Judge damping by recovery, not just by the distance.

---

## Evaluation

### MOT17
A standard pedestrian-tracking benchmark from MOTChallenge: crowded street videos with hand-labeled boxes and IDs. Scoring on it makes results comparable to published trackers.

### Anti-UAV
A dataset of drone videos with labeled drone boxes. Drones are small, fast, and often in single-target clips, so it tests different things than crowded MOT17 does.

**In Tether:** we don't run a detector on it. We feed in the *labeled* boxes, delete some on purpose, and see whether the tracker recovers (`tether-antiuav`, results in `docs/benchmarks/antiuav-dropout.md`). A track only "has" the drone if its box overlaps the label with IoU ≥ 0.5; otherwise, with one drone per clip, a track that never dies would look perfect.

**Gotcha:** the motion a tracker sees is motion *in the image*, which includes the camera's. Anti-UAV300 is filmed by a pan-tilt camera that keeps re-centering on the drone, so the drone swings back and forth across the frame in a sawtooth even when it flies straight. A constant-velocity model predicts one frame ahead fine, but coasting through a few missed frames sends it the wrong way whenever the camera snaps back.

### HOTA (Higher Order Tracking Accuracy)
A tracking metric that balances two things: how well you **detect** objects, and how well you **associate** them over time (keeping IDs right). It's the geometric mean of the two, so you can't score well by being great at one and bad at the other.

### IDF1
An F1 score for identity: of all the boxes, how many carry the *right* ID over the long run. It's very sensitive to ID switches and track fragmentation.

### ByteTrack
A well-known, simple, strong tracker, used as the baseline. Its key idea is that low-confidence detections, which most trackers throw away, still get a **second matching pass** against tracks left unmatched by the first pass. That's how it holds onto half-hidden people.

**In Tether:** both trackers get the same *unfiltered* detections, so ByteTrack keeps its second pass and the comparison stays fair. Tether has the same pass: tracks that were tracked last frame and found no detection scored ≥ 0.25 can match one scored between 0.1 and 0.25, at IoU ≥ 0.5. Those weak detections never start a track. The 0.1 and 0.5 are ByteTrack's published values, not tuned.

### Null-case test
Before trusting a metric, ask: *what score would a useless system get?* If a tracker that does nothing smart still scores well, the metric isn't measuring what you think.

**In Tether:** a tracker that never drops its one track would score 100% "same ID kept" on single-drone clips. The fix was to require IoU ≥ 0.5 with the real drone box for the ID to count.

### Train/test leakage
Scoring a model on data it was trained on. The numbers look great and mean nothing, like grading a student on the exact homework they studied. Published MOT17 trackers often use detectors trained on MOT17's train videos. Tether evaluates *on* those videos, so it can't use such a detector without a held-out split.

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

### Input size (and letterboxing)
The image size the model actually sees. A 1920×1080 frame is shrunk to fit the input (e.g. 640×640) with gray padding filling the leftover space, which is called letterboxing. Shrinking 3× turns a 60 px person into a 20 px smudge the model can't find. On MOT17, going from 640 to 1280 input more than doubled recall, while bigger models at 640 barely helped: a bigger brain doesn't help if the eyes can't see. A square input also wastes pixels on padding for wide frames, so a rectangle (e.g. 1280×736) does the same job cheaper.

### Benchmark detection set vs. shipped detector
Two separate choices. The **benchmark detection set** is what a tracker comparison runs on: detections are dumped to a file once, so it can come from a slow, strong detector, and it should, so the trackers get enough signal to be compared. The **shipped detector** is what runs live, so it has to be fast. Tether benchmarks on YOLO26n at 1280×736 and ships YOLO26n at 640 (for now), and reports MOT17 for both. See ADR 0002.

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

### Ablation
Finding out what a piece of a system is worth by changing just that piece and measuring the difference. "Ablate" means "remove", so the classic version turns one feature off; swapping one piece for another system's version works the same way. When two systems score differently, ablations tell you *which* difference matters, instead of guessing.

**In Tether:** the #22 ablations swapped one ByteTrack behaviour at a time into Tether (gate, Kalman filter, confirmation frames, assignment, matching order). Only matching order explained the gap. The ablations ran through a throwaway copy of the tracker that, with no flags, gives byte-identical output to the real one. That check proves the copy itself didn't change anything.

**Gotcha:** pieces interact, so one-at-a-time results don't simply add up. All of #22's changes together scored worse than matching order alone (950 ID switches vs. 779). After you find the winner, re-run the others *on top of it*, as the #22 write-up does.

### Tracer bullet / vertical slice
Building a thin piece of functionality that works end to end, all the way through every layer, instead of finishing one layer at a time. You get something working early, and integration problems show up early.
