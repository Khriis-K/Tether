# Fine-tune a drone detector on Anti-UAV, scored on a session-disjoint val

Every Anti-UAV result so far tests the tracker on the dataset's *labeled* boxes, with some deleted on purpose (`docs/benchmarks/antiuav-dropout.md`). Nothing in Tether has found a drone by itself: the shipped YOLO26n is pretrained on COCO, which has no drone class. The spec left "Training a drone detector" for a later stage 5. The Drone detection milestone (#41–#48) brings it in.

**Decision:** fine-tune a **separate single-class drone model, YOLO26n at 1280×736**, on Anti-UAV300 train + val, with the official val replaced by a **val made of whole held-out recording sessions**. Score it at IoU ≥ 0.5, on Anti-UAV test once at the end, on **Seraphim test as a transfer test**, and for **false alarms** on Drone-vs-Bird bird images and on Anti-UAV frames with no drone. The goal is one claim: Tether detects and tracks drones end to end, in real time at the spec's speed targets, with measured recall, ID retention and false-alarm rate.

## Evidence

Measured for this ADR from the local copy of Anti-UAV300 (visible light only: `visible.json` and `visible.mp4` per clip).

| Split | Clips | Frames | Drone labeled present | Recording sessions |
|---|---|---|---|---|
| train | 160 | 149,528 | 142,192 | 27 |
| val | 67 | 61,999 | 58,302 | 26 |
| test | 91 | 85,374 | 79,724 | 10 |

- **The official val leaks.** A recording session is the clip name minus its last two `_` fields (`20190925_101846_1_4` → `20190925_101846`). All 26 val sessions also appear in train, and within a session the clips are consecutive pieces of one video dealt out to both splits: session `20190925_101846` has train clips `1_1, 1_2, 1_3, 1_6, 1_7, 1_8` and val clips `1_4, 1_5`. At the cuts, the frames match. The last frame of train `1_3` and the first of val `1_4` are both stamped 10:20:47, with the drone in the same place, and differ by a mean of 0.78 gray levels (frames shrunk to 480×270). Val `1_5` → train `1_6` differs by 0.77 (both 10:22:07). A frame from a different session differs by 4.07. Val is "the next 30 seconds of a flight the model trained on", so it would rate any detector tuned on it too highly. (It didn't matter for the tracker work: that ran on labeled boxes, with nothing learned from pixels.)
- **Test is clean.** None of its 10 sessions appears in train or val.
- **The whole dataset is two days and one camera.** Every clip was recorded on 2019-09-25 or 2019-09-26 (train 86 / 74 clips, val 36 / 31, test 55 / 36), by a pan-tilt turret whose overlay shows the same reticle and azimuth/elevation text in every frame. Anti-UAV test can show that the detector works on new flights at that site. It can't show it works anywhere else.
- **The drones aren't tiny at 1280×736.** The shorter side of a labeled box has a median of 58 / 61 / 69 px (train / val / test, 1920×1080 source) and a 10th percentile of 38 / 38 / 40 px. Shrunk to 640 input, 1.1% / 0.2% / 0.4% of boxes are under 8 px on their short side; at 1280×736, none are.
- **Seraphim** (`lgrzybowski/seraphim-drone-detection-dataset`, CC BY 4.0) has 75,134 train and 8,349 test images, one class, all resized and padded to 640×640, merged from 23 listed sources (3 Kaggle, 19 Roboflow user projects, 1 Hugging Face). Its README says it mixes real photos, marketing images and synthetic renders. Filenames are bare numbers, so an image's source can't be told from its name.
- **Drone-vs-Bird's class labels are unusable.** This is the Roboflow YOLOv7 export (`drone-3doih/drone-76w4q`, CC BY 4.0), stretched to 640×640. Its `data.yaml` says class 0 is Drone and class 1 is Bird, but every one of train's 45,180 objects and valid's 1,782 is class 0, though 7,389 train files and 701 valid files are bird images (`BTR`, `BV`). In test, 271 of the 361 bird images (`BT`) are labeled class 0 only. The filename prefixes (`DT` drone, `BT` bird) are reliable. In the samples checked by eye, most images are close-up web photos, not distant targets against the sky.

## The choices

- **Val: whole held-out sessions.** Pool train + val (27 sessions) and hold out whole sessions as the new val: chosen at random with a fixed seed, drawn from both recording days, about 15–20% of pooled frames. The split is saved to a committed file. Test is never touched until the one final run.
- **Training data: Anti-UAV only, every 10th frame.** Neighbouring frames at 20 fps are near-copies, so a stride (default 10, half a second apart) cuts the repeats without losing motion or background change. Drone-absent frames are sampled at the same stride, so their share stays near the natural ~5%; they are the only in-domain negatives. Seraphim is never trained on. Training on it would likely make a better detector, but its test split would then stop being a transfer test, and its split is random over images rather than by session. If the transfer score is poor, training on Seraphim is the next experiment, decided by that result.
- **Model: a separate single-class drone model.** The COCO model stays the default, and every MOT17 result is untouched. A single model with COCO's classes plus drone would need COCO images mixed into training to avoid forgetting them, and a re-run of every MOT17 number. Real counter-UAS sensors run a drone-specific model anyway. The tracker ignores classes. The C++ HUD and stats panel name classes from a hard-coded COCO table, so names have to come from the model or a flag (#46).
- **Size: YOLO26n at 1280×736.** It's the shipped detector's shape (ADR 0003), so the TensorRT engine, CUDA prep and speed numbers carry over. On the Anti-UAV clip the shipped configuration is limited by video decoding at both input sizes (195.45–195.86 FPS at 1280×736 against 193.49–195.18 at 640, `docs/benchmarks/cuda-prep.md`), so the larger input costs no FPS there and keeps the small-box tail. YOLO26s at 1280×736 is detect-bound at 108.89–110.06 FPS (`docs/benchmarks/model-size.md`), under the 120 FPS target, so it's ruled out.
- **Scoring.**
  - A detection is a hit at IoU ≥ 0.5 with the labeled box, the same rule as the Anti-UAV tracking score, so detector and tracker numbers line up.
  - Models are compared on AP50. Unlike recall alone, it passes the [null-case test](../CONCEPTS.md#null-case-test): a model that outputs nothing, or the same box every frame, scores about 0.
  - Recall, precision and F1 are reported at one threshold: the one with the best F1 on the new val, frozen before test, with the tracker's 0.25 shown beside it.
  - The tracker's low-score band (0.1 up to its cutoff) must move with that threshold by a stated rule (#44).
- **Transfer test: Seraphim test, with Anti-UAV frames dropped.** Roboflow projects often repackage Anti-UAV, and the source list can't rule it out, so the test images are scanned for Anti-UAV's overlay and matches are dropped. Its numbers always carry two caveats: the mix of real, marketing and synthetic images, and the 640×640 padding (about 1.15× upscaling into a 1280×736 input).
- **False alarms: two rows.**
  - **Birds:** the share of Drone-vs-Bird `BT` test images, after near-duplicates are removed, where the detector reports any drone at the operating threshold. Ground truth comes from the filename and the labels are ignored.
  - **Empty sky:** false alarms per minute on Anti-UAV frames with the drone labeled absent, on the real camera.
  - Both are easy cases (close-up birds, empty sky). A low rate doesn't show the model can tell a distant bird from a distant drone, and the write-up has to say so.
- **Targets.** Speed is pass/fail now: the drone model in the shipped configuration must meet the spec's 120+ FPS and P99 under 16 ms on the Anti-UAV clip. There was no basis for an accuracy target before training, so accuracy targets are set from the val results **before test is run** (#47). Each one is justified by something other than the val score itself, so a target can't simply be set just under what val got. They're set in [Accuracy targets](#accuracy-targets-47) below.

## Accuracy targets (#47)

Set 2026-10-23, after the val results of #42–#46 and before any drone model was run on Anti-UAV test, Seraphim or the bird images. They're the user's decision. Each is scored once, in #48.

Every target is for `yolo26n_drone_1280x736` at the **frozen threshold 0.352** (`0.3524462878704071`, the best-F1 cut on val), on TensorRT with scores saved down to 0.01, scored by the committed tools. Each target below gives the reason for its value, the val numbers it was set against with the file each comes from, and what passing does and doesn't prove.

### Why val can't set the bar

- **Val is one site and one camera.** Its 5 sessions are new flights, but from the same turret on the same two days as train (Evidence, above).
- **Val's numbers are tuned.** Val chose the epoch, the threshold and the tracker's cutoff (`docs/benchmarks/drone-model.md`, `docs/benchmarks/drone-tracking.md`).
- **Val and test already differ with no detector at all.** On labeled boxes the tracker keeps the ID on 87.0% of val frames (`drone-tracking.md`) and 82.6% of test frames (`docs/benchmarks/antiuav-dropout.md`, the final run for #28). The reachable ceiling at the 30-frame search is 95.1% on these val clips (`drone-tracking.md`) and 94.2% on test (`docs/benchmarks/antiuav-ceiling.md`). So test is harder footage, and a target just under val would partly measure the footage, not the detector.

So each target rests on something else: a miss rate the tracker was already specified to handle, an existing spec rule, the COCO "before" row, or what the sample can resolve.

| # | Row | Target | Kind | Val (tuned) |
|---|---|---|---|---|
| 1 | Recall at 0.352, Anti-UAV test | ≥ 0.80 | pass/fail | 0.992 |
| 2 | Precision at 0.352, Anti-UAV test | ≥ 0.90 | pass/fail | 0.995 |
| 3 | AP50, Anti-UAV test | reported (1 and 2 imply ≥ 0.72) | reported | 0.995 (COCO 0.228) |
| 4 | ID kept, end to end, Anti-UAV test | ≥ 0.95 × the labeled-box row on the same clips, same run | pass/fail, relative | 89.7%, against 87.0% on labels |
| 5 | Bird images with a false drone | ≤ 8 of 86 (≤ 10%) | pass/fail, coarse | none (COCO 96.5% at 0.25) |
| 6 | Empty-sky false alarms, Anti-UAV test | ≤ 4 detections in 2.25 min (≤ 2 / min) | pass/fail, coarse | 0 in 0.50 min, partly by construction |
| 7 | Seraphim transfer | AP50 > COCO's 0.216 on the same images | pass/fail, relative | none (never run) |
| 8 | Recall on tiny and small drones | reported | reported | 0 of 8 tiny, 3 of 12 small |
| 9 | Speed, shipped configuration | 120+ FPS, P99 < 16 ms | pass/fail, met in #46 | 195.59–198.63 FPS, P99 6.46–6.49 ms |

### 1. Detector recall at 0.352: ≥ 0.80

- **Reason.** The spec's dropout test stresses the tracker by deleting 20% of the drone's boxes at random, and specifies how well it must cope (ID kept ≥ 0.95 × the ceiling). A detector with recall ≥ 0.80 keeps the tracker inside that regime. Below it, the tracker runs on worse input than it was ever specified for. For an operator, 0.80 means the drone is boxed on four frames in five.
- **Val:** 0.992 (37,961 of 38,250), per session 0.986–0.997 (`drone-model.md`). COCO, class-agnostic: 0.397 at its 0.066, 0.142 at 0.25 (`docs/benchmarks/drone-coco-baseline.md`).
- **Passing proves** the detector finds the drone on most frames of new flights at the same site. **It doesn't prove** the misses are spread out like random deletions. A detector's misses come in runs (camera snaps, the drone at the frame's edge), which hurt tracking more. Target 4 measures that.

### 2. Detector precision at 0.352: ≥ 0.90

- **Reason.** With one drone per frame, precision 0.90 is about 0.09 false boxes per frame at recall 0.80, and at most 0.11 at full recall. A false box becomes a track only after 3 hits in a row (`--confirm-frames 3`), so scattered false boxes at that rate should rarely reach the HUD (an expectation from the rule, not a measurement). The bar comes from how the tracker filters, not from val.
- **Val:** 0.995 (`drone-model.md`). COCO: 0.270 at 0.066, 0.410 at 0.25 (`drone-coco-baseline.md`).
- **Passing proves** false boxes are rare while the drone is in view. **It doesn't prove** anything about empty sky or birds (targets 5 and 6). Where val's false positives fall (a second box on the drone, or elsewhere in the frame) wasn't measured.

### 3. AP50 on Anti-UAV test: reported, no separate target

- **Reason.** Targets 1 and 2 together already imply AP50 ≥ 0.72. The precision envelope is at least 0.90 everywhere up to recall 0.80, so the area under it is at least 0.80 × 0.90. A separate bar would add a number without adding a check. AP50 stays the comparison with the COCO model's test row, which #48 dumps at the same floor.
- **Val:** 0.995, against COCO's 0.228 at the same floor and runtime (`drone-model.md`).

### 4. End-to-end ID kept: ≥ 0.95 × the labeled-box row on the same test clips

- **Rule.** In #48, `tether-antiuav --detections` runs the tracker on the drone model's test dump (cutoff 0.352, band floor 0.4 × the cutoff). In the same run it scores the same clips' labeled boxes, with nothing deleted. The target: drone ID kept ≥ 0.95 × labeled-box ID kept. The clips where the drone row beats the labeled row are reported beside it, as on val, along with the reachable ceiling.
- **Reason.** This is the spec's existing rule for dropout (≥ 95% of the ceiling, made relative in #33), applied to a real detector: the detector may cost at most 5% of what the labels give the same tracker. It's relative because test is harder for the tracker even on labels (82.6% against val's 87.0%), and that difference isn't the detector's.
- **The labeled-box row is a reference, not a ceiling.** On val the drone model beats it, 89.7% against 87.0%, because the labels lose the drone on 4 clips. On 3 they mark it absent while it's half in view. On 1 the labeled box's height jumps during a camera snap and breaks the merge's 1.5× size check. On the other 38 clips the detector is 202 frames behind in all (`drone-tracking.md`). So the target says the detector costs no more than 5% of the labels. It doesn't say the detector gets within 5% of the best possible. The reachable ceiling (94.2% on test, `antiuav-ceiling.md`) is reported beside it as a reference point, not a hard limit: it's computed from the same labels, so a detector that bridges the labels' absences can beat it too, as two val clips did.
- **Val:** drone model 89.7% (34,317 of 38,250), labeled boxes 87.0% (33,288), so 1.03 × the labeled row. By session the drone model ranges 76.3%–98.5%. COCO at 0.25: 10.9% (`drone-tracking.md`).
- **Passing proves** the tracker keeps the drone's ID on the detector's own test output about as well as on the labels. **It doesn't prove** much where the labels lose the drone: there the detector can gain frames the labels can't, which is how it gained on val. That's why the clips where it beats the labels are reported.

### 5. Bird false-alarm share: ≤ 8 of the 86 images (≤ 10%)

- **Reason.** The row asks whether the drone model calls close-up birds drones. The COCO row (96.5% at 0.25, `docs/benchmarks/drone-transfer-sets.md`) shows the metric isn't trivially 0. 10% is the coarsest bar that still separates "rarely" from "often" on this sample, so it only catches a gross failure.
- **Resolution.** One image is about 1.2 points, so the bar is a count: 8 images pass (9.3%), 9 fail (10.5%). The binomial noise at 10% on 86 images is about ±6 points (95%). The images also cluster in a few bird videos (`BT (25)` absorbed 32 near-duplicates, `BT (158)` 31), so the effective sample is smaller than 86. A 5% bar (4 images) would be decided by one or two clusters.
- **Val:** none. Val has no birds, and the drone model hasn't been run on this set.
- **Passing proves** the model doesn't mistake close-up web photos of birds for drones. **It doesn't prove** it can tell a distant bird from a distant drone, which is the operational case.

### 6. Empty-sky false alarms: ≤ 4 detections in test's 2.25 minutes (≤ 2 per minute)

- **Reason.** Test's empty sky is 2,704 frames, 2.25 minutes, in 12 clips (`drone-transfer-sets.md`). The bar has to sit several counts above 0, or one stray box decides it. 4 detections lets a single short false object pass and fails a recurring one. For scale, the COCO model at its 0.066 cut gave 17.9 per minute on val's empty sky (`drone-transfer-sets.md`), the kind of rate that would bury an operator.
- **Resolution.** One alarm is about 0.44 per minute, so the bar of 4 is 1.8 per minute, under the 2 per minute it's named by. The scorer counts detections, not frames or tracks, so one false object lasting 3 frames is 3 alarms (about 1.3 per minute).
- **Val's 0 is partly by construction.** The empty-sky rule (no labeled drone within 2 s) was sized on val labels. Its 40-frame margin is 4 frames past the drone model's furthest val drone-absent detection, at 36 frames (`drone-transfer-sets.md`, `docs/benchmarks/drone-model-absent-detections.json`). So 0 alarms on val's 0.50 minutes of empty sky, 80% of it one clip, was expected, and supports no bar. Test applies the same rule by labels and frame index only.
- **Val:** drone model at 0.352: 0 alarms in 0.50 min. At 0.25: 3 (6.0 per minute). On every drone-absent frame: 48 in 1.03 min, nearly all the drone half out of view (`drone-transfer-sets.md`, `drone-model.md`).
- **Passing proves** the model rarely fires on this site's empty sky. **It doesn't prove** a false-alert rate an operator would see elsewhere: it's a few minutes of one site's sky, the easy case.

### 7. Seraphim transfer: AP50 above the COCO model's 0.216 on the same images

The transfer test has a target.

- **Relative, not absolute.** An absolute bar has nothing to stand on. Seraphim mixes real photos, marketing images and synthetic renders from 23 merged sources. It's also padded to 640×640, so a 1280×736 input upscales it about 1.15× (Evidence, above). The COCO model is the reference: the network the drone model was fine-tuned from, scored class-agnostically on the same 8,349 images, with the same runtime and floor.
- **Reason.** The transfer test asks whether fine-tuning on one site's footage helped or hurt on drones the model never saw. Beating the model it started from is the minimum for "it learned drones, not the site". A failure triggers the next experiment named above: training on Seraphim. With 8,627 labeled drones, the sample is big enough for a real pass/fail.
- **COCO on Seraphim:** AP50 0.216, recall 0.372 and precision 0.333 at 0.25, recall 0.608 at its 0.066 (`drone-transfer-sets.md`). Its AP50 on the Anti-UAV val is 0.228 (`drone-coco-baseline.md`).
- **Val:** none for the drone model. Seraphim was never run.
- **Reported beside it, with no target:** recall and precision at 0.352, and recall by box size. 0.352 was frozen on Anti-UAV and may not suit Seraphim's score distribution.
- **Passing proves** the fine-tune didn't overfit the site so badly that it does worse than no drone training at all. **It doesn't prove** the model works on real counter-UAS footage elsewhere, since much of Seraphim isn't that.

### 8. Recall on tiny and small drones: reported, no target

- **Reason.** There are too few boxes to judge. Val has 8 tiny boxes (< 16² px) and 12 small ones (16²–32²). The drone model misses all 8 tiny and 9 of the 12 small, finding 3 of 12; COCO at 0.25 found 4 of 12 (`drone-model.md`). At 1280×736 no labeled drone in any split is under 8 px on its short side (Evidence, above), so test's tail is thin too, and a target would be decided by a handful of boxes. The write-up says plainly that finding small drones isn't shown, and Anti-UAV can't show it.

### 9. Speed: met in #46

- In the shipped configuration (TensorRT, CUDA prep, parallel) the drone model ran 195.59–198.63 FPS with P99 6.46–6.49 ms, in both rounds, against 120+ FPS and P99 < 16 ms. It's decode-bound (capture 99.7% busy). It was timed on the first val clip, `train/20190925_131530_1_1`, not the earlier speed clip, which is in test (`docs/benchmarks/drone-speed.md`).

## Consequences

- **The spec changes.** Training a drone detector is in scope as stage 5, scoped to this ADR, and the drone milestone's targets are added. The Android app, Jetson and infrared stay out of scope. Infrared, a bearing output and a low-latency mode are candidates for a later drone milestone.
- **The pretrained COCO YOLO26n, scored class-agnostically, is the "before" row** for every drone number: any detection counts as a drone.
- **End-to-end tracking is scored on the real detector's output** with the existing ID-kept rule, next to the same clips' labeled boxes (a reference, not a strict ceiling: see target 4 below). The detector's own misses are the dropout, with no synthetic deletion on top (#44).
- **Datasets and weights stay out of git**, like MOT17 and Anti-UAV. Seraphim and Drone-vs-Bird images are third-party and scraped, so none go in `docs/media` either. The split file, the training settings and seed, and the lists of dropped and deduplicated images are committed.
- **Every drone number names its model**, the same way the MOT17 rows name 640 or 1280×736, so a COCO-model number and a drone-model number can't be mixed up.
- **The claim stays narrow.** After the milestone, the honest claim is "the perception part of a counter-UAS pipeline, measured on one site's footage, with a transfer test". It isn't "a counter-UAS system".
