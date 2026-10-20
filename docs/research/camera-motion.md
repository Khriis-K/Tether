# Camera motion and target loss on Anti-UAV: what to try

- **Question:** How do we get Tether through (A) multi-frame detection gaps and (B) tentative-track churn on Anti-UAV300 visible-light, where a pan-tilt camera turns the drone's image motion into a sawtooth? Can we do it without giving the tracker images?
- **Date:** 2026-10-18
- **Scope:** Camera-motion compensation (CMC/GMC), image-free motion and gating tricks, Anti-UAV challenge methods, and lifecycle changes. This is a literature and code review only. Nothing here was run on our data. Every number below comes either from the brief (our own measurements) or from the linked source. Claims I couldn't confirm are marked **unverified**, and my own reasoning is marked **reasoning**.

## The problem, restated (numbers from the brief)

Anti-UAV300 visible-light: 20 fps, 1920×1080, one drone per clip. The pan-tilt unit keeps the drone near a crosshair, so its image motion is a sawtooth: about 28 px/frame one way for 8–10 frames, then about 50 px/frame snapping back. The background is mostly featureless sky.

- Next-frame prediction is fine: the predicted box misses IoU 0.3 on only 3.4% of frames.
- **Failure A (coasting):** prediction error is 33 / 73 / 142 px after 0.25 / 0.5 / 1 s gaps.
- **Failure B (tentative churn):** a new track needs 3 hits in a row and dies on one miss.
- **Result vs. target:** 26% ID kept with 20% of boxes deleted (target 95%). 27% of 1 s gaps recovered (target 80%).

Here's what Tether does today, from the code:

- `kalman.py`: an 8-state constant-velocity (CV) filter on (cx, cy, w, h), with noise scaled by box height. Coasting just calls `predict()` again, so the velocity is never damped and position error grows with every missed frame.
- `matching.py`: IoU Hungarian with `MIN_IOU = 0.3`, plus a center-distance matcher (≤ 4 box sizes) that is only used for tracks with one hit.
- `tracker.py`: confirmed tracks are matched first (stage 1). Tracked tracks then get a low-score pass (stage 2, IoU ≥ 0.5). Tentative tracks come next (stage 3), then the one-hit distance fallback (stage 4). A tentative track that misses is dropped. A confirmed track survives `MAX_MISSING_FRAMES = 30` missed frames as "searching", but it can only be re-matched by IoU ≥ 0.3 against its coasted prediction.

**Reasoning:** the last point is the core of failure A. A searching track can only come back if its coasted box still overlaps the detection by IoU 0.3. Drones in this dataset are small, so a 73 px error after 0.5 s is many box widths. The track can't re-match and dies, and the detection starts a new tentative track. That is where failure A turns into failure B. If a drone box is ~30 px wide (**unverified**: I don't have the box-size distribution), IoU 0.3 already fails at roughly 10–15 px of center error. That would explain why next-frame matching works but recovery after a gap doesn't.

---

## 1. Camera-motion compensation (CMC / GMC)

### 1a. BoT-SORT GMC

**What it is.** [BoT-SORT](https://arxiv.org/abs/2206.14651) estimates a 2×3 affine between consecutive frames and warps every track's Kalman state and covariance with it before association. The paper extracts "good features to track" keypoints and follows them with sparse Lucas-Kanade optical flow. It fits the affine with RANSAC. It says the correction to the velocity terms is essential at high camera speeds, and that GMC is "negligible" next to detector inference but can be slow on large images ([paper, CMC section](https://ar5iv.labs.arxiv.org/html/2206.14651)).

**What the code does** ([`tracker/gmc.py`](https://raw.githubusercontent.com/NirAharon/BoT-SORT/main/tracker/gmc.py)):

- **Methods:** `orb`, `sift`, `ecc`, `sparseOptFlow` (the default), `file`, and `none`.
  - The constructor accepts `files`, but `apply()` only dispatches `file`, so `files` silently returns identity.
  - Frames are downscaled by 2 first.
- **Keypoint methods** (`orb`/`sift`):
  - Mask out a 2% border and every detection box, so the drone itself doesn't vote on the camera motion.
  - Filter matches with a ratio test, a displacement cap of 25% of the frame, and a mean/std outlier filter.
  - Fit `cv2.estimateAffinePartial2D(..., cv2.RANSAC)`.
- **`sparseOptFlow`:** does *not* mask detections.
- **`ecc`:** runs `cv2.findTransformECC` (Euclidean, 5000 iterations, eps 1e-6). On any exception it prints a warning and keeps identity.
  - Its translation is never scaled back up after the downscale, which looks like a bug (my reading of the code).
- **Fallback:** for keypoint and flow methods, if 4 or fewer points survive, the code prints "not enough matching points" and returns **identity**.
- **`file` mode:** reads `GMC-<seq>.txt`. Each tab-separated line holds a frame field and then the six affine entries `a b tx c d ty`.

**How the affine is applied** ([`bot_sort.py` `multi_gmc`](https://raw.githubusercontent.com/NirAharon/BoT-SORT/main/tracker/bot_sort.py)):

```python
R = H[:2, :2]; R8x8 = np.kron(np.eye(4), R); t = H[:2, 2]
mean = R8x8 @ mean; mean[:2] += t
cov  = R8x8 @ cov @ R8x8.T
```

Rotation and scale are applied to every (x, y) pair in the state (position, size, velocity), and translation only to the center. Tether uses the same xywh state, so this would carry over almost line for line. One caveat: BoT-SORT's R also rotates (w, h) as if it were a vector, which is only approximately right under rotation. For a pan-tilt camera with near-zero roll, R ≈ identity, so this doesn't matter here (**reasoning**).

**Behavior on sky-dominated frames.** I found no paper that measures BoT-SORT GMC on sky frames. What the sources do say:

- **Sparse corners need texture.** Shi-Tomasi corners, ORB, and SIFT all need it. Uniform sky gives few or no keypoints, and the BoT-SORT code then falls back to identity, which is today's behavior. So the failure is **graceful but silent**: no warning reaches the tracker, and we just lose the benefit (from the code above).
- **Grid keypoints are the known workaround.** A MAV-detection paper samples **grid-based keypoints** instead of corners "for robustness in textureless regions such as sky and grassland". It fits a homography with RANSAC ([Guo et al., arXiv 2312.11008](https://arxiv.org/html/2312.11008)).
  - The same paper notes that a homography needs a planar background or a camera that "only rotates".
  - A pan-tilt camera is close to pure rotation, so a homography (or, for small angles, an affine) is the right model regardless of scene depth. **Reasoning:** for pure rotation, image motion is H = K R K⁻¹ and doesn't depend on depth.
- **ECC is intensity-based** and doesn't need corners, but it still needs gradients. On flat sky it can converge to anything or throw an exception (**unverified** for this dataset). It is also the slowest option. StrongSORT's ablation shows FPS dropping from 8.3 to 6.3 when ECC is added ([StrongSORT](https://ar5iv.labs.arxiv.org/html/2202.13514)).
- **The worst failure is wrong-but-confident, not identity.** If the only texture in a frame is the drone itself, plus perhaps a cloud edge or a crosshair overlay, an unmasked method (`sparseOptFlow`, `ecc`) will lock onto it. The estimated "camera motion" then *is* the target's motion, and compensation cancels the very motion we want to track. Masking detection boxes (as `orb`/`sift` do) is mandatory here.
  - **Reasoning plus a question for the data:** does Anti-UAV300 visible have a burned-in crosshair or OSD overlay? If so, it is static in image space and will pull the estimate toward identity. It needs masking too (**unverified**).

**Precomputing offline into a per-frame transform file.** This is well-trodden:

- BoT-SORT's `file` mode does exactly this.
- Deep OC-SORT's `cmc.py` defaults to `method="file"` and reads precomputed affines from `./cache/cmc_files/` ([Deep OC-SORT `cmc.py`](https://raw.githubusercontent.com/GerardMaggiolino/Deep-OC-SORT/main/trackers/integrated_ocsort_embedding/cmc.py)). Unlike BoT-SORT, it raises if the file is missing rather than falling back.

For Tether this keeps the tracker image-free. A separate tool reads the video once and writes `gmc/<clip>.txt` (one affine per frame, plus ideally a quality field such as inlier count). The tracker then takes one optional 2×3 matrix per `update()`.

- **Cost:** one offline pass per clip, roughly the price of an optical-flow pass (no numbers measured).
- **Online / C++ use:** the transform must then be computed live from the image, which breaks "no images" for that deployment. The tracker interface stays the same either way.
- **Precompute caveat:** if the GMC tool masks detections, it depends on the detection file. Regenerating detections means regenerating GMC, so the two need to stay in sync.

**Fit with Tether:** good. It is a pure-math warp on the state, plus one optional input per frame.

**Failure A:** potentially large. If the sawtooth is mostly camera-induced, compensated motion is close to the drone's true angular motion, which should be much smoother. Coasting a CV model in compensated coordinates then extrapolates the drone, not the gimbal controller.

- **Big caveat (reasoning):** during a gap the camera keeps moving, so the transform for every gap frame has to be applied too. That's easy offline: every frame has a transform whether or not there's a detection.
- **But** if the gimbal is tracking the drone, then when the detector misses, the gimbal's own tracker (which we don't see) may also lose it. Its motion during gaps could be odd (hold, search pattern). GMC would follow that correctly, which is a plus.

**Failure B:** indirect. Fewer lost confirmed tracks means fewer new tentative tracks.

### 1b. UCMCTrack

[UCMCTrack (AAAI 2024)](https://arxiv.org/abs/2312.08952) runs the Kalman filter on the **ground plane** through a camera homography. It matches with a "Mapped Mahalanobis Distance" and applies the same compensation parameters across a whole sequence, not per frame.

**Fit:** poor. A drone against sky has no ground-plane footpoint, and a pan-tilt camera's homography changes every frame. Its per-sequence-constant assumption is exactly wrong for a gimbal. The one transferable idea is to **match by Mahalanobis distance with the filter's covariance instead of IoU** (see 2f).

### 1c. Pan/tilt telemetry

If the PTZ's encoder angles were available, camera motion would be known exactly, with no texture needed. Pan/tilt deltas map to image shifts through the focal length.

- **Is it available?** I found **no evidence that any Anti-UAV dataset releases pan/tilt telemetry.**
  - The Anti-UAV paper mentions a pan-tilt only as a cause of motion blur at night.
  - Its annotation fields are `exist` and the box ([Anti-UAV, arXiv 2101.08466](https://ar5iv.labs.arxiv.org/html/2101.08466)).
  - The [official repo](https://github.com/ZhaoJ9014/Anti-UAV) lists boxes, attributes, and exist flags only.
  - So the absence is **unverified** but likely.
- **Prior art:** the closest is a US Air Force patent that predicts optical flow from known pan/tilt and subtracts it before a Kalman filter ([US 9,696,404 B1, via search result](https://scholar.afit.edu/patents/56)). It isn't something we can use on this benchmark.

---

## 2. Image-free approaches robust to direction reversals

### First, a reasoning check on CV vs. constant-position for a sawtooth (**reasoning**, not measured)

Take the brief's numbers: +28 px/frame for 8–10 frames, then −50 px/frame. Ramp amplitude is about 28 × 9 ≈ 250 px. The snap-back then takes about 250 / 50 ≈ 5 frames, so one cycle is about 14 frames, or 0.7 s at 20 fps. These are derived, not measured.

- **Constant velocity (CV)** extrapolates the velocity at the moment the track was lost. If that's +28 px/frame and the gap is 20 frames (1 s), CV's prediction drifts up to ~560 px. The true position has gone through at least one full cycle and is back within ±250 px. CV error therefore grows without bound with gap length. The longer the gap, the more likely it spans a reversal.
- **Constant position (CP)**, just holding the last box, has error bounded by the sawtooth's peak-to-peak range (~250 px), whatever the gap length. But at short gaps it's worse than CV: 5 frames into a +28 ramp, CP is ~140 px off. Our measured CV-ish error at 0.25 s is 33 px.
- **So CV wins short gaps and CP wins long ones.** The crossover is somewhere around a half-period (~5–7 frames, ~0.3 s). Our measured errors (33 / 73 / 142 px) are already much smaller than naive CV would give at 1 s. Our filter isn't naive: its velocity estimate likely averages over the sawtooth somewhat. So the crossover could be later. **We don't know; we need to measure prediction error vs. gap length for CV, CP, and damped-CV on val.**
- **Damped velocity** (v ← αv each coasted frame) interpolates between the two. It behaves like CV for the first few frames and like CP after ~1/(1−α) frames. That's the shape we want.
- **A dataset-specific prior:** the pan-tilt is *trying to keep the drone at the crosshair*, so image position mean-reverts toward a fixed point. A mean-reverting (Ornstein-Uhlenbeck-like) model pulling toward the crosshair would beat CP for long gaps. But it's a strong, Anti-UAV-specific assumption that would be meaningless on MOT17. Flag it as an overfitting risk.

### 2a. Velocity damping while coasting

**Summary.** Multiply the velocity components by α < 1 on each `predict()` with no matching update. In effect this adds a decay to F's velocity block while the track is searching. There's no dedicated MOT paper for this as a named method. ByteTrack does something related: it zeroes the height velocity of non-tracked tracks before predicting (`mean_state[7] = 0` for non-tracked tracks, [ByteTrack `byte_tracker.py`](https://raw.githubusercontent.com/ifzhang/ByteTrack/main/yolox/tracker/byte_tracker.py)). The damped idea itself is standard (an Ornstein-Uhlenbeck velocity / Singer-style model, **unverified**: no primary source fetched).

- **Cost:** trivial. About two lines in `kalman.py`, no images, and one hyperparameter α to tune on train/val.
- **Fit:** perfect.
- **A:** directly targets it: it caps the CV overshoot.
- **B:** indirect.

### 2b. Constant-position fallback / IMM (CV + CP)

**Summary.** An Interacting Multiple Model (IMM) filter runs several Kalman filters in parallel, for example CV and CP or random walk. It mixes them by model probabilities that update from each model's measurement likelihood ([APL Tech Digest overview](https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V22-N04/22-04-Genovese.pdf)). [IMM-MOT (arXiv 2502.09672)](https://arxiv.org/pdf/2502.09672) uses CV, constant-acceleration, and turn models for 3D MOT.

I found no 2D tracking-by-detection paper using CV + CP specifically (**unverified** whether one exists).

- **Catch (reasoning):** during a gap there are no measurements, so IMM's model probabilities freeze at their last values. The mixture keeps coasting with whatever blend it had. IMM helps with *switching while observed* (it would track the snap-back faster), but on its own it does not decide "switch to CP because we've been blind for a while". A simpler "after k missed frames, zero the velocity" rule targets A more directly.
- **Cost:** moderate. Two filters per track and a mixing step, no images. It's more code than 2a for an unclear extra gain.
- **Fit:** fine.
- **A:** partial (see catch).
- **B:** none.

### 2c. Adaptive process noise (Q growing with time since update)

**Summary.** Inflate the process noise Q, or the covariance P, faster while a track coasts. Uncertainty then reflects how lost the track really is. This only helps if association *uses* the covariance, through Mahalanobis gating. Tether's IoU matcher ignores P entirely, so inflating Q alone changes nothing about matching. Tether's P already grows each `predict()`; it's just unused.

The related **NSA Kalman** from GIAOTracker, used in StrongSORT, scales the *measurement* noise by detection confidence, R̃ₖ = (1 − cₖ)Rₖ ([StrongSORT §3](https://ar5iv.labs.arxiv.org/html/2202.13514)). That's measurement-side: low-confidence detections pull the state less. It isn't about coasting.

- **Cost:** trivial in code, but only useful together with 2f.
- **A:** yes, if paired with covariance-aware gating.
- **B:** no.

### 2d. OC-SORT: ORU, OCM, OCR

[OC-SORT](https://arxiv.org/abs/2203.14360) argues that KF error accumulates fast while a track goes unobserved. In its simplified model, position-noise variance grows as 2T²σ² over T untracked steps ([paper](https://ar5iv.labs.arxiv.org/html/2203.14360)).

- **ORU (Observation-centric Re-Update).** When a lost track is re-associated, it rebuilds a *virtual trajectory* between the last observation before the gap and the new one. It then re-runs predict/update along it, so the drifted velocity is replaced. The code interpolates **linearly** (constant-speed) in `unfreeze()` ([`kalmanfilter.py`](https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/kalmanfilter.py)).
  - **For us:** it fixes the state *after* recovery. It doesn't help *achieve* recovery. It would stop a post-gap velocity spike, which may matter on a sawtooth.
  - Cheap, no images. **A:** small. **B:** none.
- **OCM (Observation-Centric Momentum).** Adds a direction-consistency term (weight 0.2) between a track's observed direction (over Δt = 3 frames) and the direction to each candidate detection. The cost is scaled by detection score ([`association.py`](https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/association.py)).
  - **For us (reasoning):** with one drone per clip there's rarely a competing detection to disambiguate. And the sawtooth *reverses direction by design*, so OCM would penalize correct matches right after a snap-back. Likely neutral to harmful here.
- **OCR (Observation-Centric Recovery).** After the main pass, match unmatched tracks' *last observed box* (not the prediction) to unmatched detections by IoU ≥ `iou_threshold` ([`ocsort.py`](https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py)). The paper pitches it for objects "stopping or being occluded for a short time".
  - **For us:** this *is* a constant-position fallback inside matching, so it's a cheap hedge against CV overshoot.
  - **Limit:** the code only applies it to unmatched tracks in the same frame against their last box, still by IoU. With small boxes and a 250 px sawtooth, IoU with the last box fails quickly too. A distance-based OCR would be more useful (see 2f).
  - **A:** moderate. **B:** none.

Our repo has history here: issue #25 found "OC-SORT pieces don't hold up" on MOT17 (commit `395e3b5`). That was MOT17. Anti-UAV's failure is different, but don't expect OCM to help.

### 2e. Expanded / buffered IoU (C-BIoU) and friends

[C-BIoU (WACV 2023)](https://arxiv.org/abs/2211.14317) grows each box by b × (w, h) on every side before computing IoU. It matches in two cascades, first a small buffer, then a large one for what's left. It uses no Kalman filter: velocity is the average motion over the last 2–5 frames. Buffers were grid-searched over 0.1–0.7 ([paper](https://ar5iv.labs.arxiv.org/html/2211.14317)). The authors note it does worse than plain IoU under heavy detection noise (40% noise).

- **For us:** a buffer helps small, non-overlapping boxes match, but it scales with box size, not time-since-seen. To cover a 142 px error on a ~30 px box (**unverified** size), b would have to be huge. Then it's effectively center distance. Better to gate on distance directly, growing with time since seen.
- Related cost functions:
  - DIoU / GIoU / CIoU and center distance are in OC-SORT's `association.py`.
  - BoostTrack++ adds Mahalanobis and shape similarity plus a soft BIoU ([arXiv 2408.13003](https://arxiv.org/abs/2408.13003)).
  - Hybrid-SORT adds height and confidence as "weak cues" ([AAAI 2024](https://arxiv.org/abs/2308.00783)). I couldn't confirm its exact "height-modulated IoU" formula from the abstract (**unverified**).
- **Cost:** low. **Fit:** good. **A:** partial. **B:** small.

### 2f. Re-association gate that grows with time since last seen

**Summary.** For *searching* tracks, add a stage that matches by center distance with a threshold growing with `frames_since_seen`, or by Mahalanobis distance using the filter's P, which already grows each coasted frame. This is DeepSORT's original gating idea. StrongSORT describes DeepSORT's use of the Mahalanobis distance as a motion gate ([StrongSORT §2](https://ar5iv.labs.arxiv.org/html/2202.13514)).

Tether already has `match_by_distance` (≤ 4 box sizes). It's only used for one-hit tentative tracks. Extending it to searching tracks with a radius like `r0 + k·frames_since_seen` (pixels or box sizes) is a few lines.

- **Reasoning:** this is the most direct fix for "the track survives in memory for 30 frames but can never re-match because IoU 0.3 against a coasted small box is hopeless". Combined with 2a (damping), the predicted center stays inside the sawtooth range. The growing gate covers the rest.
- **Cost:** low. No images, reuses existing code.
- **Fit:** perfect.
- **A:** high (it is the recovery mechanism). **B:** high indirectly: every recovered track is one tentative track not spawned.
- **Risk:** a wide gate can grab a false positive (bird, cloud edge) and keep a wrong ID alive. Watch the false-match rate on val, i.e. track boxes far from GT while "tracked".

---

## 3. Anti-UAV / small-target papers and challenge entries

Nearly all Anti-UAV challenge entries are **single-object trackers (SOT)**. They get the first-frame box and look for one target with local search plus global re-detection, usually with image features. Common patterns:

- **Global re-detection after loss is the biggest lever.**
  - UTTracker (CVPRW 2023, 2nd in the 3rd challenge) combines background alignment, global detection, multi-region local tracking, and dynamic small-target detection. The challenge report gives its cumulative ablation as base 53.7 → +background alignment 57.5 → +global detection 65.9 → +multi-region local tracking 68.3 → +dynamic detection 68.8 ([3rd challenge report, arXiv 2305.07290](https://ar5iv.labs.arxiv.org/html/2305.07290); [paper page](https://mlanthology.org/cvprw/2023/yu2023cvprw-unified)).
  - Global detection gave the biggest jump, and background alignment (i.e. camera-motion compensation) gave a real but smaller one.
  - I couldn't read the PDF (403), so how its alignment works and when it switches local→global is **unverified**.
- **2nd challenge, BIT_OITS (SiamSTA):**
  - Local search near the last position when the camera is "static", global re-detection when lost.
  - It decides "static camera" by tracking 5–30 Shi-Tomasi points with LK flow and forward-backward checks: static after five stable frames.
  - It switches local/global based on agreement with frame-difference / background-subtraction change detection ([2nd challenge report, arXiv 2108.09909](https://ar5iv.labs.arxiv.org/html/2108.09909)).
- **2nd challenge, COLA Try and JNU:**
  - COLA Try aligns backgrounds with optical flow + warping, then background-subtracts to find motion.
  - JNU verifies the local result and falls back to a sliding-window global search ([same report](https://ar5iv.labs.arxiv.org/html/2108.09909)).
- **3rd challenge winner (Colalab):** uses **tracking-by-detection**: a strong detector, a "Simple Tracker" with cascading rules, a video-clip re-scorer, and a frame-difference motion model. SOT was only post-processing ([arXiv 2305.07290](https://ar5iv.labs.arxiv.org/html/2305.07290)).
- **MAV detection (Guo et al.):** global and local detectors with a switcher that goes back to global search only after the local one "fails for a while" (30 frames as an example). Background compensation uses grid keypoints + RANSAC homography for textureless sky ([arXiv 2312.11008](https://arxiv.org/html/2312.11008)).
- **Other benchmarks:**
  - [Anti-UAV410 (T-PAMI 2023)](https://doi.org/10.1109/TPAMI.2023.3335338) (IR only).
  - [AntiUAV600 / evidential detection-tracking collaboration (arXiv 2306.15767)](https://arxiv.org/abs/2306.15767): "global UAVs detection and local UAVs tracking".
  - [DUT Anti-UAV (IEEE T-ITS)](https://arxiv.org/abs/2205.10851) (visible light; detector + tracker fusion).
  - I didn't read their full texts. Details of their loss handling are **unverified**.
- **PTZ telemetry:** none of these use it, and none of the datasets appear to provide it (see 1c).

**How this translates to a MOT-by-detection design like Tether:**

- **"Global re-detection" is free for us.** The detector already runs on the full frame every frame. What we lack is the *association* side: re-attaching a full-frame detection to the lost track instead of minting a new ID. That's lifecycle (§4) and gating (2f), not detection.
- **"Local vs. global search switching"** maps to staged gating: tight IoU gate while tracked, growing distance gate while searching, and (single-target prior) an unbounded gate after N frames.
- **"Background alignment"** maps to GMC (§1), precomputed offline.
- **Verification / re-scoring** (Colalab's video checker, JNU's verification) need images or temporal features. They're out of scope for an image-free tracker, but could live in the detector stage.

---

## 4. Lifecycle tweaks

### 4a. Tentative tracks surviving a miss (failure B)

How the reference implementations handle tentative tracks:

| Tracker | Tentative track on a miss | Source |
|---|---|---|
| DeepSORT | deleted on first miss (`mark_missed`) | [track.py](https://raw.githubusercontent.com/nwojke/deep_sort/master/deep_sort/track.py) |
| ByteTrack | unconfirmed track removed if unmatched next frame | [byte_tracker.py](https://raw.githubusercontent.com/ifzhang/ByteTrack/main/yolox/tracker/byte_tracker.py) |
| BoT-SORT | same: `mark_removed()` on unmatched unconfirmed | [bot_sort.py](https://raw.githubusercontent.com/NirAharon/BoT-SORT/main/tracker/bot_sort.py) |
| OC-SORT | **not deleted**: `hit_streak` resets, but the tracker lives until `time_since_update > max_age` (30) | [ocsort.py](https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py) |
| Norfair | **survives one miss**: `hit_counter` starts at 1, +2 per hit, −1 per frame, dies below 0. Promotion when counter > `initialization_delay` (default 7 with `hit_counter_max=15`) | [norfair tracker.py](https://raw.githubusercontent.com/tryolabs/norfair/master/norfair/tracker.py) |

Tether matches DeepSORT/ByteTrack/BoT-SORT. On a detector that drops frames, the Norfair-style leaky counter is the obvious change: it tolerates intermittent hits while still rejecting one-off false positives. Note OC-SORT also *outputs* only tracks with `hit_streak >= min_hits` (3), so its "confirmed" output is still streak-based.

- **Cost:** trivial.
- **Fit:** perfect.
- **B:** high.
- **Risk:** more false tracks confirmed. On a one-drone dataset that hurts precision only if the detector has persistent false positives.

### 4b. Re-identifying a lost confirmed track instead of a new ID

- **Lost tracks stay matchable.** ByteTrack and BoT-SORT keep lost tracks for `buffer_size = int(frame_rate / 30 * track_buffer)` frames and match them in the *first* association pass. Matches reactivate with the old ID (`re_activate(..., new_id=False)`) ([BoT-SORT](https://raw.githubusercontent.com/NirAharon/BoT-SORT/main/tracker/bot_sort.py), [ByteTrack](https://raw.githubusercontent.com/ifzhang/ByteTrack/main/yolox/tracker/byte_tracker.py)). Tether already does this (searching tracks are in stage 1).
  - **Detail:** the buffer is scaled by frame rate. Tether's `MAX_MISSING_FRAMES = 30` is 1.5 s at 20 fps, but would be 1 s at 30 fps. Worth making it time-based.
- **The gap is the gate, not the buffer.** As above, IoU against a coasted prediction is too strict for small fast targets.
- **Norfair's `reid_hit_counter_max`** keeps dead objects around for a separate ReID matching window ([norfair](https://raw.githubusercontent.com/tryolabs/norfair/master/norfair/tracker.py)). This is the same idea as a "lost buffer with a different matcher".
- **Single-target prior (dataset-specific, overfitting risk).** With exactly one drone per clip, a new high-confidence detection anywhere in the frame while the only confirmed track is searching is most likely the same drone. A rule like "if there is exactly one searching track and no tracked track, match it to the best high detection regardless of distance" would directly attack both A and B.
  - It's a strong prior that's false in general MOT (MOT17, multi-drone scenes).
  - It would inflate Anti-UAV numbers without making Tether a better tracker.
  - If used, keep it behind an explicit option such as `max_targets=1`. Report results with and without it, and never tune it on test.

---

## Summary table

A = coasting through missed detections, B = tentative-track churn.

| Approach | Compute | Needs images? | Complexity | Fit with Tether | Targets A | Targets B |
|---|---|---|---|---|---|---|
| GMC, precomputed to per-frame file (grid KLT + RANSAC, detections masked) | offline pass per clip; ~free in tracker | offline only | medium (tool + state warp) | good (one optional affine per frame) | high *if* sky has enough texture; else identity (no gain) | indirect |
| GMC online (BoT-SORT style) | per-frame flow | **yes** | medium | breaks design | as above | indirect |
| UCMCTrack ground-plane | low | needs camera params | high | poor (no ground plane in sky) | low | none |
| PTZ telemetry | ~free | no | low | good | high | indirect |
| Velocity damping while coasting | ~free | no | trivial | perfect | high (caps overshoot) | indirect |
| IMM CV + CP | 2× filter | no | medium | ok | partial (probabilities freeze in gaps) | none |
| Q growing with time since seen | ~free | no | trivial | needs Mahalanobis gate | only with 2f | none |
| NSA Kalman (R scaled by score) | ~free | no | trivial | good | small | none |
| OC-SORT ORU | small | no | low | good | small (post-recovery only) | none |
| OC-SORT OCM | small | no | low | good | likely harmful (sawtooth reverses) | none |
| OC-SORT OCR (last-box IoU) | small | no | low | good | moderate | none |
| Buffered IoU (C-BIoU) | small | no | low | good | partial | small |
| Distance/Mahalanobis gate growing with time since seen | small | no | low (reuse `match_by_distance`) | perfect | **high** | high (indirect) |
| Leaky tentative counter (Norfair-style) | ~free | no | trivial | perfect | none | **high** |
| Time-based lost buffer | ~free | no | trivial | perfect | small | small |
| Single-target prior (`max_targets=1`) | ~free | no | low | ok (dataset-specific) | high | high |

---

## Ranked shortlist: what to try first

1. **A growing re-association gate for searching tracks** (center distance, radius growing with `frames_since_seen`; reuse `match_by_distance`). The track already survives a gap in memory but can't re-match through IoU 0.3 on a small coasted box. This is the missing link between A and B.
   - **How we'd know it failed:** on val, "1 s gaps recovered" doesn't move toward 80%, or the rate of *wrong* re-associations jumps. Wrong means a track box matched while IoU with GT < 0.1. Check both, not just ID-kept.
2. **Velocity damping while coasting** (decay v each missed frame; optionally zero it after k frames). It caps the CV overshoot the sawtooth guarantees, keeps the predicted center inside the drone's range, and makes gate #1 tighter for free.
   - **How we'd know it failed:** plot val prediction error vs. gap length for undamped, damped, and CP. If damped isn't at or below the min of the other two at 0.25 / 0.5 / 1 s, the α is wrong or the reasoning is. The null case: if CP alone beats everything, the velocity estimate is useless during gaps.
3. **Leaky tentative counter** (Norfair-style: survive one miss, confirm on net hits). It directly targets failure B: a dropout-prone detector breaks 3-in-a-row streaks.
   - **How we'd know it failed:** on val, confirmed false tracks per clip rise (tracks never overlapping GT), or ID-kept doesn't move because churn was really caused by lost confirmed tracks (fixed by 1–2), not by tentative deaths. Measure what fraction of new IDs come from tentative deaths vs. confirmed-track expiry *before* changing it.
4. **Offline GMC to a per-frame affine file** (grid keypoints + LK + RANSAC, detection boxes masked, log inlier count). It's the only option that removes the sawtooth's cause rather than hedging it, and it keeps the tracker image-free. It's ranked fourth only because it's more work and might do nothing on featureless sky.
   - **How we'd know it failed:** compute the fraction of val frames with too few inliers (identity fallback). If it's high, GMC is a no-op. Also check that the compensated drone trajectory is actually smoother: its frame-to-frame velocity variance should drop. If it doesn't, the estimate is locking onto the drone or an overlay.
5. **Single-target prior, behind an explicit flag.** It likely gives a big Anti-UAV gain for little code, but it's dataset-specific. Try it last, and only as a labeled upper bound.
   - **How we'd know it's misleading:** it helps Anti-UAV but the same flag hurts or is meaningless on MOT17. Report it separately and never fold it into the default config.

Measure items 1–3 separately before combining. They interact: damping changes where the gate is centered. All tuning goes on train/val, and the test split stays reserved.

---

## Unverified / open questions

- **Frame rate: answered, 20 fps.** The Anti-UAV paper says "25 FPS" ([arXiv 2101.08466](https://ar5iv.labs.arxiv.org/html/2101.08466)), but every one of the 318 `visible.mp4` files in the download reports 20.0 fps, and each clip's label count matches its frame count.
- **Drone box sizes: answered, larger than assumed.** Over all labeled visible frames (all splits), box area is about 2,400 / 7,400 / 16,000 px² at the 10th / 50th / 90th percentile, roughly 49 / 86 / 127 px square. The ~30 px assumption above is too small, so IoU tolerates more center error than estimated there; the conclusions about coasting still need measuring on val.
- **Telemetry: answered, on screen.** No telemetry file ships with Anti-UAV300, but every visible frame has the pan-tilt azimuth ("方位") and elevation ("俯仰") burned in at the top left, to 0.01°, updating every few frames. Not used yet (#32).
- **Sky texture: answered, too little.** 76.2% of val frame pairs fall back to identity, and compensation lowers the drone's velocity variance by only 5.8% (#32, `docs/benchmarks/antiuav-camera-motion.md`).
- **Crosshair overlay: answered, yes.** The visible frames have a burned-in, image-fixed crosshair (bracket lines around the center), seen in rendered frames of `20190925_134301_1_2`. A GMC tool would need to mask it.
- **UTTracker.** Its background-alignment method and local/global switching rule are unverified (PDF returned 403).
- **IMM.** I found no 2D MOT paper using CV + CP specifically.
- **Damped velocity.** The Singer / Ornstein-Uhlenbeck velocity model as the formal basis isn't verified from a primary source here.
- **Hybrid-SORT.** The exact "height-modulated IoU" formula was not confirmed.
- **ECC.** Behavior on flat sky (diverge vs. exception) is unverified for this data.
- **Is the sawtooth all camera?** It may be entirely camera-induced, or partly real drone motion. Only GMC (or telemetry) can tell. If the drone's true motion is itself erratic, GMC won't smooth it.
