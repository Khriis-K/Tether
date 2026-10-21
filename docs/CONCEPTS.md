# Concepts

A running list of the ideas behind Tether, in plain English. Each entry says what the thing is, then how it shows up in this project. Add to it as new ideas come up.

(This is a learning notebook. The canonical project vocabulary for agents lives in `CONTEXT.md`, if and when it exists.)

The stories behind these ideas (the problems we hit, how we found the causes, and what we learned) are in [LESSONS.md](LESSONS.md).

**Contents:** [Tracking basics](#tracking-basics) · [Matching & motion](#matching--motion) · [Evaluation](#evaluation) · [Detection & runtimes](#detection--runtimes) · [Performance](#performance) · [Python vs. C++](#python-vs-c) · [Display](#display) · [Design & testing ideas](#design--testing-ideas)

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

### Leaky counter (tried, shelved)
A softer confirmation rule for new tracks: instead of needing N hits in a row, a tentative track keeps a running count that goes up on each hit and down on each miss, and is confirmed when the count reaches N. A track can then ride out a missed frame, while a one-off false detection still never builds up enough hits. [Norfair](https://github.com/tryolabs/norfair) uses one.

**In Tether:** tried for #31 and not merged (code on the branch `issue-31-tentative-miss-experiment`). A tentative track survived one miss and confirmed on net hits (hits minus misses). It cut the new IDs handed out on Anti-UAV by a fifth, but ID kept didn't move (85.3% → 85.2% at 20% deletion), and MOT17 at 640 lost 0.46 HOTA. See `docs/benchmarks/tentative-miss.md`.

**Gotcha:** if a miss costs a whole hit, surviving one is worth at most one frame. Hit, miss, hit, hit, hit confirms on the same frame as a fresh track started at the second hit. And fewer IDs handed out isn't the same as a better score: the [duplicate-track merge](#duplicate-track-merge) already gave the drone back its old ID, so the IDs this saved were ones nothing scored.

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

**Gotcha:** the reach is far wider than the gate, because the drone often reappears 5–10 box sizes from its prediction. That is safe in a crowd (the cap) and on a single-target benchmark (nothing to get wrong), but not in between. In a sparse scene, a lone person who walks out can hand their ID to a same-size newcomer who walks in elsewhere. A scripted webcam scene showed the search gate already does this for gaps of 5–29 frames, because a box-size gate around a big, close-up box spans most of the frame; the merge adds the case where the newcomer arrives on the very next frame. The [edge rule](#edge-rule-tried-shelved) was tried against this and shelved.

**Gotcha:** center distance ignores size. A big box walking in front of a smaller, occluded track can have its center inside that track's gate, which IoU would never allow. Tether's gate also requires the heights to be within 1.5 times of each other, against the track's last *observed* box: a coasted prediction's size drifts with the size velocity (MOT17-09 had predictions 1211 px tall, and one with negative height).

### Edge rule (tried, shelved)
A track whose last box touched the frame edge most likely left the scene, so it shouldn't hand its ID to whoever appears next. The rule blocks only the distance-based re-matches (the gate and the merge), so IoU still keeps a briefly missed object at the edge.

**In Tether:** tried for #37 and not merged (code on the branch `issue-37-edge-rule-experiment`). It fixed the scripted walk-out/walk-in swap, but it cost Anti-UAV val about 1 point of ID kept and regressed MOT17-04 at 1280×736. On real data the swap it targets was rare. See `docs/benchmarks/edge-rule.md`.

**Gotcha:** touching the edge isn't the same as leaving. On MOT17-04 the rule blocked two correct re-matches of people standing partly cut off at the top edge. And without appearance (below), the same person walking back in looks exactly like a newcomer, so the rule gives both a new ID.

### Camera-motion compensation (GMC) (tried, not feasible)
When the camera pans, every object's image motion includes the camera's. Global motion compensation estimates that camera motion between frames, usually as an affine fit to how background points moved (optical flow plus RANSAC). It then warps each track's predicted box by it, so the filter models only the object's own motion. BoT-SORT does this online. Tether would read it from a precomputed per-frame file to stay image-free.

**In Tether:** not built. #32 checked feasibility first on Anti-UAV val: 76.2% of frame pairs had too few textured background points to fit, and compensation lowered the drone's velocity variance by only 5.8% overall. On the few textured clips it halved the variance, so the method works; the sky just has nothing to register against. See `docs/benchmarks/antiuav-camera-motion.md`. The pan-tilt angles burned into each frame don't rescue it either: they update every ~11 frames, slower than the camera's ~5-frame jerks (`docs/benchmarks/antiuav-telemetry.md`).

**Gotcha:** it fails *confidently*, not loudly. A point on flat sky has no gradient, so optical flow leaves it in place and says "no motion", and a forward-backward check agrees. That's the **aperture problem**: you can only see motion where the image has texture to move. Static overlays (a crosshair, on-screen text) do the same. So a 0% fallback rate can mean every frame confidently estimated "the camera didn't move". Check the result, not just how many points survived: here, whether the tracked object's motion actually got smoother.

### Appearance re-identification (re-ID)
Telling objects apart by what they look like, usually from an embedding a neural network computes for each box crop, instead of only by where the box is and how it moves. It's what lets a tracker recognize the same person coming back after leaving the frame, which no motion rule can do.

**In Tether:** not built. Tether is motion-only, which keeps it fast and simple to port to C++. #37 named re-ID as the alternative if motion rules couldn't stop sparse-scene swaps, and kept it out of scope.

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

**Gotcha:** the motion a tracker sees is motion *in the image*, which includes the camera's. Anti-UAV300 is filmed by a pan-tilt camera that keeps re-centering on the drone, so the drone swings back and forth across the frame in a sawtooth even when it flies straight. A constant-velocity model predicts one frame ahead fine, but coasting through a few missed frames sends it the wrong way whenever the camera snaps back. Measured on the textured clips: the gimbal holds still, then jumps about 550 px in roughly 5 frames, then holds again (`docs/benchmarks/antiuav-telemetry.md`).

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

### Reachable ceiling
The best score any system could get under a rule it can't break, worked out from the labels alone. If the ceiling is below the target, no tuning will reach the target.

**In Tether:** a track that searches longer than its window without a detection is forgotten, so an Anti-UAV clip where the drone leaves the labels for longer than that forces a new ID. Splitting each clip at those absences and keeping only its longest piece gives the best possible "ID kept" (`id_kept_ceiling`, `tools/antiuav_ceiling.py`). With the 30-frame default it's 94.0% on test at 20% deletion, under the old flat 95% target, so the target is now relative: ID kept ≥ 95% of the ceiling (`docs/benchmarks/antiuav-ceiling.md`, #33).

**Gotcha:** it's an upper bound, not a forecast. It ignores everything else that loses IDs (confirmation, the IoU rule, bad matches). And with no search limit it's 100% by construction, because nothing else can go wrong on a single-drone clip; the cost of a longer window only shows up in crowds, where a dead track can be revived on the wrong person.

### Oracle experiment (upper bound)
Give one stage of the system information it couldn't really have, and measure how much the score goes up. That's the most that stage could ever add, so it tells you whether working on it is worth it before you do.

**In Tether:** the single-target prior (`tether-track --single-target`, #34) knows each Anti-UAV clip has one drone. When the only confirmed track is searching, it takes the nearest detection anywhere, with no gate. At 20% deletion on val that takes ID kept from 85.3% to 93.3% against a 96.5% [ceiling](#reachable-ceiling), so perfect re-association is worth 7.9 of the 11.2 missing points. Most of the rest is frames where the drone jumps away from a track that's still tracked (`docs/benchmarks/antiuav-single-target.md`).

**Gotcha:** an oracle's number is never a result. It's off by default, never counts toward a target, and never ships: on general footage "one object" is false, and with one person in view it would hand a lost ID to anyone who walks in. Also check it actually fires where you run it: on MOT17 its condition never holds, so it scores the same as the default.

### Noise floor (how big a difference is real)
The size of change you get from things that shouldn't matter. A difference between two rows smaller than that can't be credited to what you changed. Measure it by changing something irrelevant and seeing how far the score moves.

**In Tether:** the same YOLO26n model run by ONNX Runtime on the CPU and by TensorRT on the GPU gives detections that agree to 0.00032 in score, yet the same tracker's MOT17 score moved by up to 0.39 HOTA, 0.56 IDF1 and 12–40 ID switches. A threshold turns a tiny score difference into a different decision, and the decision compounds over frames. So in the model-size table (#18), gaps under about 0.4 HOTA, 0.6 IDF1 or 40 ID switches are treated as noise (`docs/benchmarks/model-size.md`).

**Gotcha:** two runs of the same thing can match exactly, as these do (the scoring is deterministic for a given detection file), and still hide this noise. The noise only shows up when an input changes by an "irrelevant" amount.

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

In Tether: both sit behind the same `Detector` interface, and `tether-pipeline --runtime onnxruntime|tensorrt` picks one. ONNX Runtime runs on the CPU (its CPU provider), TensorRT on the GPU, so the comparison is CPU against GPU, not two GPU runtimes.

### TensorRT engine
What TensorRT turns an ONNX model into: a compiled plan (`.engine` or `.plan` file) holding the layers it fused and the GPU kernels it timed as fastest. Building one took 67–90 s for YOLO26n here; after that the pipeline only loads it.

In Tether: built once per machine with `trtexec --onnx=models/yolo26n.onnx --saveEngine=models/yolo26n.engine --noTF32`, and loaded from next to the model (README, "TensorRT runtime").

Gotcha: an engine is tied to the GPU model, the TensorRT version and the input size it was built with. That's why engines stay out of git, and why each model × input size needs its own. An engine from another TensorRT version is refused when it loads (`tether-pipeline` then says to rebuild it); one built on a different GPU may load with only a warning, so treat engines as per-machine build output.

### Strongly typed network (precision from the model)
TensorRT 11 takes each layer's precision from the model's own types, instead of picking lower precision on its own (`trtexec` no longer has `--fp16`). A float32 ONNX model gives a float32 engine. TF32, a float32 mode of the GPU's matrix units that keeps fewer mantissa bits, is allowed by default (`--noTF32` turns it off).

In Tether: engines are built with `--noTF32`, so TensorRT runs the same plain float32 model as ONNX Runtime, and the speedup is from compilation and the GPU, not from lower precision. With TF32 on, 0.66% (640) and 1.05% (1280×736) of MOT17-04's detections came back with scores more than 0.02 off ONNX Runtime's (up to 0.315), enough to fail the agreement check at 1280×736. With it off, the worst score difference was 0.00032, and TensorRT's detect time changed by −0.054 to +0.341 ms a frame (`docs/benchmarks/tensorrt-runtime.md`). Running FP16 would need an FP16 ONNX export.

Gotcha: TF32 is on by default and the build log only says "Precision: Strongly Typed", so it's easy to miss. And even plain float32 isn't bit-identical across runtimes: kernels add numbers in a different order, so boxes and scores differ in the last digits and need a tolerance (see [Detection agreement](#detection-agreement-between-runtimes)).

### Detection agreement (between runtimes)
How to say two runtimes give "the same" detections when the numbers can't match exactly. Each detection one runtime finds looks for a partner in the other: same class, boxes overlapping by a high IoU, scores close. Among the boxes that overlap enough, take the closest score, or two near-identical boxes on one object can fail against the wrong twin. Checked both ways, so neither runtime can drop or invent boxes unnoticed.

In Tether: `tether-runtime-compare` checks detections scoring at least 0.25 (the tracker's threshold), and a partner needs IoU ≥ 0.95 and scores within 0.02. The runtimes agree when at least 99% of those detections have one. The tolerance was set before the first comparison on the real model.

Gotcha: only count detections above a threshold, and let partners come from below it. A box scoring 0.251 in one runtime and 0.249 in the other is the same detection, and a check that thresholds both sides would call it missing.

### CUDA kernel
A function that runs on the GPU across thousands of threads at once. It's a natural fit for per-pixel work like resizing and color conversion.

### Frame prep (preprocessing)
Turning a raw camera frame into what the model expects: resize to the input size, reorder color channels (OpenCV uses BGR, models usually want RGB), and scale pixel values (e.g. 0–255 down to 0–1).

### Host vs. device memory
"Host" means CPU RAM and "device" means GPU memory. Copying between them costs time. Doing frame prep on the GPU and handing the result *straight* to the detector skips a round trip.

### Input size (and letterboxing)
The image size the model actually sees. A 1920×1080 frame is shrunk to fit the input (e.g. 640×640) with gray padding filling the leftover space, which is called letterboxing. Shrinking 3× turns a 60 px person into a 20 px smudge the model can't find. On MOT17, going from 640 to 1280 input more than doubled recall, while bigger models at 640 barely helped: a bigger brain doesn't help if the eyes can't see. A square input also wastes pixels on padding for wide frames, so a rectangle (e.g. 1280×736) does the same job cheaper.

**In Tether:** on all 7 MOT17 videos, n goes from 33.7% recall at 640 to 54.6% at 1280×736. A bigger model adds less but isn't worthless: at 640, s reaches 40.0% and m 41.5%; at 1280×736, 58.0% and 60.3% (`docs/benchmarks/model-size.md`, #18).

**Gotcha:** ADR 0002's "bigger models barely help" came from two videos. On all seven, s at 640 is worth 5.07 HOTA over n at 640. Input size is the bigger lever, not the only one.

### Benchmark detection set vs. shipped detector
Two separate choices. The **benchmark detection set** is what a tracker comparison runs on: detections are dumped to a file once, so it can come from a slow, strong detector, and it should, so the trackers get enough signal to be compared. The **shipped detector** is what runs live, so it has to be fast. Tether benchmarks on YOLO26n at 1280×736 and ships YOLO26n at 640 (for now), and reports MOT17 for both. See ADR 0002. #18 recommends shipping YOLO26n at 1280×736 too: on TensorRT in parallel mode it meets the speed targets on the Anti-UAV clip (124.82–126.18 FPS), so the two would be the same model (`docs/benchmarks/model-size.md`).

---

## Performance

### Throughput vs. latency
- **Throughput:** how many frames per second come out (FPS).
- **Latency:** how long one frame takes from camera to screen (end-to-end delay).

They're different. A parallel pipeline can raise FPS without making any single frame faster, and can add delay per frame, sometimes a lot.

**In Tether:** the timing log stores each step's start and end time, not just how long it took, so both can be measured from one log. Frame time is the gap between frames finishing; delay is from a frame being captured to it being drawn. In the sequential pipeline they're nearly the same number. In the parallel one they split apart, and only timestamps can tell. Measured (#14, one-frame queues, video files): FPS up 1.23–1.57×, and delay up from about 37 ms to 147–159 ms. That's a [Little's law](#littles-law) effect (`docs/benchmarks/parallel-pipeline.md`).

**Gotcha:** the measured delay starts when the program has the frame. The camera's own sensor-to-USB delay comes before that and isn't counted, so a true camera-to-screen number is higher.

### Tail latency (P99 / worst 1%)
The frame time that 99% of frames beat. Averages hide the occasional slow frame, but the viewer sees it as a stutter. A pipeline averaging 8 ms with a 40 ms P99 *feels* worse than a steady 12 ms.

### Nearest-rank percentile
One way to compute a percentile: sort the values and pick the one at position ⌈p × n⌉. The answer is always a value you actually measured. NumPy's default instead *interpolates* between neighbors, which can give a frame time no frame ever had. On small samples the two disagree a lot.

**In Tether:** `tether-timing` uses nearest rank, and so does the C++ `p99` behind the live stats panel, which works it out over the last 300 frames. With under 100 frames, P99 is simply the worst frame.

### Sequential vs. parallel (pipelined) stages
- **Sequential:** capture → prep → detect → track → draw, one frame at a time. Simple to debug, but every stage waits on the others.
- **Parallel:** each stage runs on its own thread and works on a different frame at the same moment, like an assembly line. Throughput becomes limited by the *slowest single stage*, not the sum of all of them.

**In Tether:** `tether-pipeline --pipeline parallel` (#14) runs capture, prep, detect and track on four threads, and draws on the main one, because a GLFW window has to be drawn on the thread that made it. Frames pass through FIFO queues, so the tracker still sees them in order: recordings with the tracks drawn came out byte-identical to sequential's on MOT17-09 and the Anti-UAV clip. The frame time settled at the detect time (about 33 ms), so the gain is capped at sequential frame time over detect time: 1.23× on the Anti-UAV clip, 1.57× on MOT17-04, where JPEG decoding had more to hide. On the webcam it took the camera's dropped frames from 129 to 0 in 300, with just 0.2 ms to spare per frame.

**Gotcha:** overlapped steps compete for the same cores. Decoding ran up to twice as slow in parallel, because ONNX Runtime keeps its own threads busy during detect. It didn't matter here because decoding wasn't the bottleneck, but it would with a faster detector.

### Bounded queue & backpressure
The buffers between pipeline stages have a fixed maximum size. If a slow stage falls behind, its input queue fills up and the faster stage before it has to wait. That pushback is **backpressure**. Without a limit, a fast producer would pile up frames forever, using more and more memory and adding more and more delay.

**In Tether:** `BoundedQueue` (`cpp/src/tether/bounded_queue.hpp`) holds one frame between each pair of stages. Closing a queue is also how the parallel pipeline stops. A stage that reaches the end of the video closes its output, and the stages after it finish what's queued and stop too. A quit or an error closes every queue, which wakes any stage waiting on one.

**Gotcha:** a bigger queue doesn't make anything faster once the slow stage always has its next frame ready. It only adds waiting. Going from two frames to one kept FPS and cut delay by about two detect times (#14).

### Little's law
In any steady system, *items inside = arrival rate × time each spends inside*. Turned around for a pipeline: **delay = frames in flight × frame time**. A full pipeline running at 30 FPS with 5 frames in flight has about 167 ms of delay, however fast each step is.

**In Tether:** on a video file, the parallel pipeline's capture and prep run ahead of the detector until every queue before it is full, so about five frames are in flight (one held by capture, one queued, one in prep, one queued, one in detect). Five times a 33 ms frame predicts about 165 ms, and the measured delay was 147–159 ms. With two-frame queues, two more frames waited and delay rose by about two frame times. The webcam's delay was lower (83 ms) because capture can't run ahead of the camera (`docs/benchmarks/parallel-pipeline.md`).

**Gotcha:** for a live camera, the fix for delay is to drop old frames rather than queue them, that is, to always detect the newest one. That trades smoothness for freshness, and Tether doesn't do it yet.

### Amdahl's law
Speeding up one part of a job only helps in proportion to how much of the total time that part takes. If tracking is 5% of frame time, making it 10× faster saves at most about 4.5%.

**In Tether:** this is why the spec reports the tracking-step speedup (expected to be large) and the whole-pipeline speedup (expected to be smaller, because the detector dominates) as separate numbers. Measured: tracking is 10–25× faster in C++, but the whole pipeline only 1.08–1.20×, because the detector is the same ONNX Runtime call in both languages and takes 66–83% of a C++ frame. Python frame time over Python detect time gives the most C++ could ever win with that detector: 1.27–1.76× depending on the video, so the spec's 2× needs a faster detector (`docs/benchmarks/cpp-pipeline-speed.md`).

### Dropped frames (webcam)
A camera produces frames at its own fixed rate whether or not the program is ready. If the pipeline takes longer than one camera period per frame, the camera keeps capturing and some frames are never read: they're dropped. A live demo has to keep up or it stutters.

**In Tether:** the C++ webcam source counts drops from the camera's own frame timestamps. A gap of about *k* periods since the last frame means *k* − 1 frames dropped, rounded so ordinary jitter doesn't count. Each timing-log line carries the count, and `tether-timing` totals it. A camera that reports no frame rate or timestamps gets no count at all, rather than a misleading 0. On the laptop camera (30 FPS), the sequential C++ pipeline read 300 frames and the camera dropped 69 more: its detect step alone (33.7 ms) is longer than the 33.3 ms period.

**Gotcha:** wall-clock time can't see drops. The camera driver buffers a few frames, so a slow pipeline still gets a frame instantly on many reads, from the buffer, and looks fine until the buffer overflows. Then frames vanish in bursts (here, three at a time, about every 13 frames). Only the camera's timestamps show where the gaps are.

### Microbenchmark (timing one step alone)
Timing a single piece of code in isolation, away from everything around it, so its own speed is all you measure. The usual rules: time only that code, run it several times and keep the median, and do exactly the same work on both sides of a comparison.

**In Tether:** `tether-parity` times the tracking step alone. A fresh tracker runs `update` over a whole detection file, with file reading and writing left out, 5 times per language, keeping the median. The C++ tracking step is 14.1× faster on MOT17 and 35.0× faster on Anti-UAV (`docs/benchmarks/cpp-tracking-speed.md`).

**Gotcha:** a microbenchmark can be fooled in both directions. A timer that starts in the wrong place still returns a positive number, a few nanoseconds. A test only caught that once it demanded a time no real run could beat. And a big speedup on one step says little about the whole program ([Amdahl's law](#amdahls-law)).

### Profiling (Nsight Systems)
Measuring where time actually goes before optimizing. Nsight Systems is NVIDIA's timeline profiler: it shows CPU threads, GPU kernels, and memory copies side by side.

**In Tether:** the rule is to profile *before* writing the CUDA kernel, to prove frame prep is a real bottleneck instead of assuming it. #15 did, with the `nsys` CLI on the parallel pipeline (`docs/benchmarks/nsight-profile.md`). At the shipped 640 input, prep was never the slowest stage: capture was. Inside TensorRT's detect, the GPU was idle most of the time, waiting on [kernel launches](#kernel-launch-overhead-and-cuda-graphs). `tools/nsight_steps.py` turns a capture's SQLite export into per-step and per-part numbers.

**Gotcha:** the profiler changes what it measures. Tracing every CUDA call added 0.67–1.49 ms to TensorRT's detect step, enough to make detect the slowest stage on the Anti-UAV clip when, without the profiler, capture is. So the profile shows what's *inside* a step, and runs without the profiler show how long each step takes.

### NVTX (named ranges on a profiler timeline)
NVIDIA's tiny annotation API: push a named range when a piece of code starts and pop it when it ends, and Nsight Systems draws it as a labelled bar on that thread's timeline. With no profiler attached, the calls do nothing.

**In Tether:** `StepRange` in `pipeline.cpp` marks capture, prep, detect, track and draw, in both pipeline modes. It's only compiled in with TensorRT (which brings the CUDA toolkit's header-only NVTX).

**Gotcha:** libraries use NVTX too. TensorRT marks every layer, 27,383 NVTX events in a 100-frame trial run, so profile with `--nvtx-domain-include=default` to keep only your own ranges.

### Kernel launch overhead (and CUDA graphs)
Every GPU kernel is started by a call from the CPU, and each call costs microseconds. A model made of hundreds of small kernels can spend longer *launching* than the GPU spends running them, and the GPU sits idle in between. A CUDA graph records the whole sequence of launches once and replays it with one call.

**In Tether:** TensorRT's YOLO26n engine launches 217 kernels per frame at 640. Profiled, the CPU took 5.81–6.10 ms to launch them while the GPU computed for 1.47–1.53 ms. `trtexec` on the same engine took 4.04–4.06 ms to enqueue without a graph and 0.028–0.029 ms with one. The launch time also explains TensorRT's wide P99: in the slowest 1% of frames, the kernels took the same time and the launches took about twice as long. A [CUDA graph](#cuda-graph-capture-once-replay-each-frame) took the launch to 0.407–0.438 ms (#40).

**Gotcha:** "faster GPU" won't fix a launch-bound model. Detect barely grew from 640 to 1280×736 for exactly this reason: the kernels got slower, the launches didn't.

### CUDA graph (capture once, replay each frame)
A recording of GPU work. Put a stream into capture mode, issue the work as usual (kernel launches, copies), end the capture, and CUDA hands back a graph of everything that was issued instead of running it. Instantiate the graph once, and each `cudaGraphLaunch` runs the whole recording with one call. It replays the same kernels with the same arguments and the same memory addresses, so the buffers have to stay where they were.

**In Tether:** `TensorRtDetector` captures one `enqueueV3` on its own stream at start-up and replays it every frame; the copies in and out stay outside the graph (#40, `docs/benchmarks/cuda-graph.md`). At 640, detect went from 5.10–6.37 ms to 2.32–2.97 ms with byte-identical detections. `--no-cuda-graph` turns it off. If capture fails, it warns and launches directly.

**Gotcha:** capture has rules. It needs a stream of your own (not the default stream), and work that can't be recorded fails the capture, so run the engine once *before* capturing: TensorRT does its one-off setup on the first run. Always end the capture, even when the recorded call failed, or the stream stays stuck in capture mode. And a graph only saves *launch* time: in parallel mode, where FPS is set by the slowest stage, a 2× faster detect raised FPS 1.02–1.10×, because decoding was slower. Profile with `nsys --cuda-graph-trace=node` to see the kernels inside a graph; by default Nsight shows the graph as one block.

### Busy share (finding the bottleneck stage)
A stage's total working time divided by the run's wall time. In a pipeline where each stage has its own thread, the slowest stage is busy nearly all the time, and the others wait. FPS is set by that stage, so speeding up any other one changes nothing.

**In Tether:** at 640 with TensorRT, capture was busy 99.2–99.8% and prep 22.2–64.7%, so a faster prep couldn't raise FPS. At 1280×736 on the Anti-UAV clip, prep was busy 99.1%: the bottleneck. `busy_share` in `tools/nsight_steps.py` computes it from a timing log or a profile, and `tether-model-compare` names each row's slowest stage. In the model-size table (#18) the bottleneck moves with the model: capture for n and s at 640, prep for n at 1280×736, detect for m at 640 and for s and m at 1280×736. So the same CUDA-prep work helps one row and does nothing for its neighbour.

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

**In Tether:** that's why the Python-vs-C++ check requires *matching* outputs (same IDs and statuses, boxes within a tiny tolerance), not bit-identical ones. `tether-parity` checks this to 1e-6 px. On all 462 real detection files we have, the C++ port's boxes differ from Python's by at most 2.6e-9 px (`docs/benchmarks/cpp-tracking-speed.md`).

### Bit-exact ports (matching NumPy's float32 math)
Sometimes "within a tolerance" isn't good enough, and you want both languages to produce the very same bytes. Then the port has to copy not just the formula but the number types and the rounding rules.

**In Tether:** the C++ frame prep and detector write detection files byte-identical to Python's (1000 Anti-UAV frames and 525 MOT17 frames, checked with `cmp`). Three details made that work. Python's `round()` rounds halves to even (2.5 → 2), so C++ uses `std::nearbyint`, not `std::round`. Scaling pixels to [0, 1] divides in float32, as NumPy does. And box decoding is done in float32 too. Under NumPy 2's rules, a float32 array combined with a plain Python number stays float32, so `(x1 - pad) / scale` rounds to float32 at every step. Doing it in double gave 337.66667683919275 where Python gives 337.6666564941406.

**Gotcha:** both shortcuts pass every test that uses round numbers (scale 0.5, scale 1). Only a test with an awkward scale like 0.3, or a size that lands exactly on .5, catches them.

### Tie-breaking (why the port copies SciPy's solver)
When two different answers are exactly as good, which one an algorithm returns depends on details like the order it scans its inputs. That's not a bug, but two implementations that both find "an optimal answer" can still disagree.

**In Tether:** Hungarian matching can have several equally cheap pairings, most often in symmetric scripted scenes. If the C++ tracker picked a different one than Python, the IDs would swap and stay swapped. So the C++ solver is a line-by-line translation of SciPy's `linear_sum_assignment`, including the order it scans columns in and its rule for ties, and it's tested against SciPy's answers on tie-heavy matrices. Every other loop (tracks in ID order, detections in file order) follows the Python too.

**Gotcha:** a textbook Hungarian implementation would pass every test that has one best answer, and could still drift from Python on real data, wherever a tie came up.

### Shortest round-trip number printing
Printing a decimal number with the fewest digits that still read back to exactly the same value: `0.1` rather than `0.1000000000000000055511151231257827`. Python's `repr` and `json.dumps` do this, and C++17's `std::to_chars` can too.

**In Tether:** the C++ file writer prints numbers exactly as Python's `json.dumps` does, including when it switches to scientific notation (`1e-05`, `1e+16`), so track files from the two trackers look the same and a text diff shows only real differences.

**Gotcha:** "shortest" doesn't settle the notation. C++'s shortest output prints 10^16 as `1e+16` in one mode and `10000000000000000` in another, and Python picks by its own rule (scientific below 1e-4 and from 1e16 up), so the port has to apply Python's rule itself.

---

## Display

### Golden-ratio colours
A way to give each new item its own colour without knowing how many items there will be. Step the hue (the position around the colour wheel) by the golden ratio, about 0.618 of a turn, for each new ID. Because the golden ratio is the "most irrational" number, the hues never line up into bunches: any first N of them stay spread around the wheel, and back-to-back IDs land far apart. An even split (1/N apart) is better, but only if you know N up front, which a tracker never does.

**In Tether:** `track_hue` in `cpp/src/tether/hud.cpp`. A test checks that for every N up to 60, the closest two of the first N hues are at least 0.3/N apart. The hues also skip a narrow band around the accent orange, so no track looks like "searching".

**Gotcha:** colour alone still isn't enough. Some IDs land close together by chance, and some viewers can't tell red from orange. That's why status is shown by line style (see the next entry).

### Redundant encoding (not by colour alone)
Showing one piece of information two ways at once, so it still reads if one channel fails, for example for colour-blind viewers or on a washed-out projector. It's a standard accessibility rule: never use colour as the only signal.

**In Tether:** a track's status is shown both by colour and by line style. Tracked objects get solid corner brackets. The prediction of where they'll be next is a thin dashed box. A searching object is a thick, longer-dashed box in the accent colour, with a "missing N" label. A test compares the searching box's ink with a prediction's, so "thicker" can't quietly drift away.

### Immediate-mode GUI (Dear ImGui)
A way to build a user interface where the program redraws the whole UI every frame from its current data, instead of creating widgets once and updating them. Each frame you call `Text(...)`, `ProgressBar(...)` and so on, and the library turns those calls into triangles for the GPU. There's no widget state to keep in sync, which suits a display whose numbers change every frame. Dear ImGui is the common C++ library for it, popular for game and tool overlays.

**In Tether:** the stats panel (`cpp/src/tether/panel.cpp`) is one function, `draw_panel`, called every frame with the latest stats, tracks and settings. `GuiScreen` (`gui_screen.cpp`) runs it in a GLFW window through OpenGL, with the video as a texture beside it. Because ImGui only produces vertex lists, the tests draw the panel in a context with no window and check the vertex colours: no accent orange unless a track is searching, and no colour outside the palette.

**Gotcha:** ImGui's defaults (grey theme, its own font, rounded windows) show through anywhere you don't override them. The panel pushes its own colours for every element it uses, and a test catches any stray colour.

### Vsync (waiting for the monitor's refresh)
Holding each new frame until the monitor starts its next refresh, so the picture never tears. The cost is that showing a frame can block for up to one refresh period (16.7 ms at 60 Hz), and the frame rate can't go above the refresh rate.

**In Tether:** the `--show` window turns it off (`glfwSwapInterval(0)`). With it on, the wait would land inside the `draw` step's time, inflating it and the frame time, and the FPS number couldn't go past 60. A timed pipeline should measure its own work, not the monitor's schedule.

### Image-sequence frame rate
A folder of numbered images (how MOT17 ships its videos) has no frame rate of its own: it's just pictures. When OpenCV opens one through FFmpeg it can report FFmpeg's default of 25 FPS, which is made up. A PNG sequence in our tests reported nothing at all.

**In Tether:** `--record` saves at the source's frame rate. Trusting the 25 made MOT17-09, really 30 FPS, play back slow (`docs/spec-verify/issue-12.md`). So an image sequence now reports "can't tell" (0), and the recording falls back to 30. The true rate is in each MOT17 video's `seqinfo.ini`; a few are 14 or 25 FPS and would still record at the wrong speed.

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
