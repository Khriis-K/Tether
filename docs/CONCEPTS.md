# Concepts

A running list of the ideas behind Tether, in plain English. Each entry says what the thing is, then how it shows up in this project. Add to it as new ideas come up.

(This is a learning notebook. The canonical project vocabulary for agents lives in `CONTEXT.md`, if and when it exists.)

**Contents**
- Ideas: [Tracking basics](#tracking-basics) · [Matching & motion](#matching--motion) · [Evaluation](#evaluation) · [Detection & runtimes](#detection--runtimes) · [Performance](#performance) · [Python vs. C++](#python-vs-c) · [Design & testing ideas](#design--testing-ideas)
- Case studies: [Problems we hit, and how we met them](#problems-we-hit-and-how-we-met-them)

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

### Growing search gate
A searching track's prediction drifts further from its object every frame it coasts, so a fixed match threshold eventually can't reach it. A growing gate widens the allowed center distance with each missed frame (`r0 + k·frames_since_seen`, in box sizes).

**In Tether:** `--search-gate` / `--search-gate-growth`, 1.5 + 0.25 by default, with the nearest-track cap below. Uncapped, it raises Anti-UAV 1 s gap recovery from 31% to 88%, but on MOT17 every setting that helps costs HOTA and ID switches.

**Gotcha:** the gate is a recall-vs-precision trade, and the cost only shows up where there are other objects to grab. With one target per clip, a wide gate can't pick the wrong box, so Anti-UAV alone will always say "wider is better". A crowded benchmark is the real test.

### Nearest-track cap
A fixed gate can't be right for both a lone object and a crowd. Capping it at a fraction of the distance to the nearest other track adapts it to the scene: alone, the full gate applies; in a crowd, it shrinks to the space no other track is closer to. At a fraction of one half, two tracks' gates can't overlap.

**In Tether:** `--search-gate-cap`, 0.5 by default, measured to other *confirmed* tracks only. It takes the gate's MOT17 cost from about 500 extra ID switches to none, and costs Anti-UAV about 1 point of ID kept.

**Gotcha:** "alone" means alone among the tracker's tracks, not alone in the world. On Anti-UAV there is one drone, yet the cap binds in about 30% of gate checks, because the tracker's own duplicate IDs of the drone sit nearby.

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

### Tracer bullet / vertical slice
Building a thin piece of functionality that works end to end, all the way through every layer, instead of finishing one layer at a time. You get something working early, and integration problems show up early.

---

## Problems we hit, and how we met them

The ideas above, in action. Each case is a real problem from this project, in roughly the order we hit it: what went wrong, how we found the cause, what we did, and the lesson that carries over to other projects. The numbers come from the linked write-ups; go there for the full tables.

### A fast object got a new ID every frame (#2)
**Problem:** an object moving 50 px/frame from its very first frame got 20 IDs in 20 frames.

**Cause:** a brand-new track has no velocity, so it predicts the object will stay put. The next detection doesn't overlap that prediction (IoU = 0), so the track never gets the match that would teach it a velocity. It's a chicken-and-egg problem.

**What we did:** tracks seen exactly once get a second chance, matched by center distance instead of IoU (see *Center-distance fallback*). 20 IDs became 1. Tuning the filter's *Process noise* mattered as much: the pedestrian value (1/160) broke the speed-up test, and high noise on box *size* turned one partial detection into an inside-out predicted box.

**Lesson:** write the hard scene as a test first. A scripted 20-frame scene found this in seconds, with no video or model. See ADR 0001.

### The first MOT17 score was low, and the tracker wasn't to blame (#6)
**Problem:** HOTA 27.8 on the first MOT17 run.

**How we found it:** the tracker's recall (32.1%) sat right at the detector's own recall (about 34%). A tracker can't keep an ID on a person the detector never found. Splitting recall by person height showed the pattern: people under 100 px were almost never found.

**What we did:** tested the cheapest explanation, resolution, before anything else. 1920×1080 frames shrunk to 640 make people 3× smaller. At 1280 input, 50–100 px people on MOT17-04 went from 0% found to 61%. A 3× bigger model at 640 still found 0%. We then split the *Benchmark detection set* from the *Shipped detector*, and later moved to a 1280×736 rectangle that costs 42% less than the square.

**Lesson:** before tuning a stage, check it isn't capped by the stage in front of it. Comparing a stage's output to its input's ceiling is a quick way to check. See ADR 0002.

### Our own precision numbers were wrong
**Problem:** detector precision looked worse at higher resolution than it was.

**Cause:** our scoring script counted detections on static people, people on vehicles, and reflections as false alarms. MOT17's official rule ignores those. That understated precision by 2–9 points.

**What we did:** rescored with the official rule, which is to match against *all* ground-truth boxes first, then drop matches to the ignored classes. Some conclusions held and one flipped: on MOT17-02, native resolution went from behind to ahead on precision.

**Lesson:** a quick homemade metric has to follow the official rules, or it measures something else. Check it against the official tool on one case before trusting it. See `docs/handoffs/2026-10-18-mot17-precision-ignore-classes.md`.

### Tether made ~470 more ID switches than ByteTrack (#22, #23)
**Problem:** on the same detections, Tether made 1270 ID switches to ByteTrack's 800.

**First guess, wrong:** ByteTrack's low-score second pass. A run of ByteTrack with that pass turned off showed it was worth only 0.2 HOTA, well under a fifth of the gap.

**How we found it:** ablations. We swapped one ByteTrack behaviour into Tether at a time: gate, Kalman filter, confirmation frames, reported box, assignment. Only one moved the numbers: **matching order**. Tether matched tentative and confirmed tracks in one pass, so a new tentative track (often from a duplicate box) could steal the detection of the confirmed track under it, then get confirmed under a new ID. That happened 847 times.

**What we did:** confirmed tracks match first, and tentative tracks get the leftovers. ID switches went from 1270 to 779 and HOTA from 35.6 to 40.9. Adding the low-score pass afterward (#26) was a smaller, consistent gain.

**Lesson:** when two systems differ, change one thing at a time and measure each change. The obvious suspect was the wrong one.

### The "ID kept" score could be faked by doing nothing (#8)
**Problem:** Anti-UAV clips have one drone each. A tracker that never drops its one track would score 100% "same ID kept", even if the box drifted off the drone.

**What we did:** a frame counts only if the track's box overlaps the labeled drone at IoU ≥ 0.5. Frames whose detection we deleted on purpose aren't scored, so 20% deletion doesn't cap the score at 80%. A tracker that outputs nothing on a gap gets no rate instead of a perfect one.

**Lesson:** this is the *Null-case test*. Ask what a useless tracker would score before you trust a metric.

### Tether found the drone but kept losing its ID (#8, #29)
**Problem:** with nothing deleted, some track held the drone on 93% of frames, but the main ID held it on only 37.6%. One 1,000-frame clip used 81 IDs.

**How we found it:** one-frame predictions were fine and missed only 3.4% of frames. But each miss cost the ID for good. The track went searching and coasted off on its old velocity, so IoU could never re-match it, and the drone's detection started a new track. The misses come from the camera: it's a pan-tilt unit that keeps snapping back to re-center the drone, so the drone moves in a sawtooth in the image (see the *Anti-UAV* gotcha).

**What we did:** a *Growing search gate* lets searching tracks re-match by center distance, with a gate that widens each missed frame. On val, ID kept at 20% deletion went from 30.5% to 61.1%, and 1 s gap recovery from 31.3% to 88.4%.

**Lesson:** fix the stage where the loss happens. Prediction was already near the limit of a constant-velocity model, and recovery was the weak stage.

### The fix for drones broke crowds (#29, #35)
**Problem:** every gate setting that helped Anti-UAV hurt MOT17. The one we picked added 513 ID switches.

**Why one benchmark couldn't see it:** with one drone per clip, a wide gate can never grab the wrong object, so Anti-UAV always says "wider is better". MOT17's crowds are the only place a false match shows up.

**What we did:** four variants failed, including a later stage order, distance per axis, and a gate scaled by speed. The fix was the *Nearest-track cap*: the gate can't reach past half the distance to the nearest other confirmed track. Checking video by video then turned up a second bug: on MOT17-09, a big box walking past an occluded person landed inside its gate. A height check fixed that. The gate is now on by default. MOT17 ended slightly better than before (731 ID switches, down from 748), and Anti-UAV kept most of its gain (59.9% ID kept, 86.6% of 1 s gaps).

**Lesson:** a change tuned on one benchmark needs a second benchmark that can show its cost. Check per video too, because totals can hide one video getting worse.

### Better predictions didn't mean more re-matches (#30)
**Problem:** damping the velocity while coasting lowered the average prediction error, but without the gate it *lowered* recovery (0.5 s gaps: 54.4% → 44.1%).

**How we read it:** IoU matching is all-or-nothing within about a box size. A lower average miss doesn't mean more predictions land within a box. We haven't measured that, so it's a guess. The null check also told us something: freezing the box (damping 0) did as well as anything at 1 s, so past about half a second the velocity estimate is useless here.

**What we did:** kept damping in the code but off by default. With the gate and cap on, it adds ID switches on MOT17 at 640.

**Lesson:** judge a change by the outcome you care about, not a stand-in for it. Always run the null version.

### Experiments changed while they ran
**Problem:** the package is an editable install, so editing `python/tether` during a long benchmark run changed the code the run was using. Parallel runs sharing a results folder could also overwrite each other's files.

**What we did:** froze the baseline in a separate git worktree (`git worktree add ../Tether-baseline HEAD`), and gave every run its own `--work-dir`. We tune on Anti-UAV *val* and keep *test* for one final run, so the test numbers don't leak into the tuning (see *Train/test leakage*).

**Lesson:** when you compare A and B, make sure each one stays exactly as it was until its run finishes.

### Still open: the drone holds two IDs at once (#36, #33)
**Where it stands:** ID kept at 20% deletion is 59.9% on val, against a 95% target. The main suspect is duplicates. After a missed jump, a new track starts on the drone and gets confirmed while the old one is still searching, and then the two trade the drone back and forth.

**A wrinkle:** even a perfect tracker that gives up after 30 frames can reach only 94.2% on test with nothing deleted, because the drone sometimes leaves the labels for longer. The target sits above what's reachable, and deciding what to do about that is #33.

**Lesson so far:** write down the ceiling next to the target. If the ceiling is below the target, no amount of tuning will reach it.
