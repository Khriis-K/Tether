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
- **Targets.** Speed is pass/fail now: the drone model in the shipped configuration must meet the spec's 120+ FPS and P99 under 16 ms on the Anti-UAV clip. There was no basis for an accuracy target before training, so accuracy targets are set from the val results **before test is run** (#47). Each one is justified by something other than the val score itself, so a target can't simply be set just under what val got.

## Consequences

- **The spec changes.** Training a drone detector is in scope as stage 5, scoped to this ADR, and the drone milestone's targets are added. The Android app, Jetson and infrared stay out of scope. Infrared, a bearing output and a low-latency mode are candidates for a later drone milestone.
- **The pretrained COCO YOLO26n, scored class-agnostically, is the "before" row** for every drone number: any detection counts as a drone.
- **End-to-end tracking is scored on the real detector's output** with the existing ID-kept rule, next to the labeled-box ceiling. The detector's own misses are the dropout, with no synthetic deletion on top (#44).
- **Datasets and weights stay out of git**, like MOT17 and Anti-UAV. Seraphim and Drone-vs-Bird images are third-party and scraped, so none go in `docs/media` either. The split file, the training settings and seed, and the lists of dropped and deduplicated images are committed.
- **Every drone number names its model**, the same way the MOT17 rows name 640 or 1280×736, so a COCO-model number and a drone-model number can't be mixed up.
- **The claim stays narrow.** After the milestone, the honest claim is "the perception part of a counter-UAS pipeline, measured on one site's footage, with a transfer test". It isn't "a counter-UAS system".
