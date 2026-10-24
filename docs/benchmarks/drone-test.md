# Drone detection: the one test run, scored against the targets

Run once, 2026-10-23, for #48, following [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md). It's the first time the drone model `yolo26n_drone_1280x736` has seen Anti-UAV test, Seraphim test or the Drone-vs-Bird bird images. The targets were set from val in #47, before this run ([ADR 0004, Accuracy targets](../adr/0004-drone-detector-on-anti-uav.md#accuracy-targets-47)). Nothing was tuned after the numbers came in, and no test run was repeated.

Raw results: the `drone-test-*.json` files beside this one, named in each table. `drone-test-inspection.json` holds the one-off counts behind the error descriptions: empty-sky alarms by clip, the missed frames on three clips, Seraphim's large-drone images and the bird images with an alarm.

**Result: 2 of the 6 accuracy targets scored here pass, and 4 fail.** Speed (target 9) was met in #46 and isn't re-run here.
- **Pass:** recall (0.884) and precision (0.975) at the frozen 0.352 on Anti-UAV test.
- **Fail:**
  - end-to-end ID kept is 0.911 × the labeled-box row, against ≥ 0.95;
  - 19 of 86 bird images get a false drone, against ≤ 8;
  - 190 empty-sky alarms in 2.25 min, against ≤ 4;
  - Seraphim AP50 is 0.164, under the COCO model's 0.216.

The drone model is far better than the COCO model it started from on Anti-UAV test: AP50 0.915 against 0.402, and ID kept 75.3% against 15.4% at 0.25. It doesn't transfer to Seraphim, and on this site's footage it fires on a night-time lamp and on some birds.

## What was frozen, and how it was checked first

Recorded before any test dump was made.

| Setting | Value |
|---|---|
| Model | `models/yolo26n_drone_1280x736.onnx` (epoch 19, #43), engine `models/yolo26n_drone_1280x736.engine` (#46's rebuild, `trtexec --noTF32 --skipInference`, TensorRT 11.4.0.106) |
| "Before" model | `models/yolo26n_1280x736.onnx` (pretrained COCO, class-agnostic), engine `--noTF32` |
| Detector | C++ `tether-pipeline --runtime tensorrt --prep cuda`, sequential, scores saved down to **0.01**. Binaries from #46's build of commit `d5885b7`; nothing under `cpp/`, `python/` or `tools/` changed between it and this run's base `0a6bb98` |
| Threshold | **0.3524462878704071** for the drone model (best F1 on val, #43). The COCO rows use its own val best-F1 cut, 0.06577478349208832, and 0.25 |
| Tracker | C++ `tether-track --min-score 0.3524462878704071`. The band floor is 0.4 × the cutoff (the default ratio). Every other setting is the default. The COCO rows run the same tracker at 0.25 and at 0.066 (#44's configurations) |
| Sets | `splits/antiuav.json` test (the official 91 clips, 79,724 drone-labeled frames), `splits/antiuav-empty-sky.json` (2,704 test frames, 2.25 min), `splits/seraphim-test.json` (8,349 images, 8,627 drones), `splits/drone-vs-bird-bt.json` (86 images) |

**Dry run on val, before test.** The exact commands below were first run on the val and COCO sets that already had committed numbers. Every one reproduced them exactly:
- **Drone val redump:** byte-identical to #46's dump on all 42 clips.
  - `tether-drone-score` at 0.352 gives `drone-model-rebuilt-engine.json`'s AP50 and rows.
  - `--empty-sky` gives 0 alarms at 0.352 and 3 at 0.25, as in `drone-empty-sky-val-drone.json`.
  - `tether-antiuav --detections` with the C++ tracker keeps 34,317 frames on the detector and 33,288 on the labeled boxes, the same on every clip as `drone-tracking-drone-cpp-at-0.352.json`.
- **COCO redumps of Seraphim, the birds and val:** byte-identical to #45's and #43's dumps. The image scorer gives Seraphim AP50 0.216 and 83 of 86 birds at 0.25, as committed.

**The test run.** The four dumps ran in one pass with no crash, in this order: the drone model on Anti-UAV test, the COCO model on Anti-UAV test, then the drone model on Seraphim and on the birds. All four took 24 min 28 s. Then they were scored once.

## Results against the targets

Every row is `yolo26n_drone_1280x736` at 0.352 unless it says COCO (`yolo26n_1280x736`, class-agnostic).

| # | Row | Target | Result | Sample | Verdict | File |
|---|---|---|---|---|---|---|
| 1 | Recall, Anti-UAV test | ≥ 0.80 | **0.884** (70,499 of 79,724) | 91 clips, 10 sessions | **PASS** | `drone-test-drone.json` |
| 2 | Precision, Anti-UAV test | ≥ 0.90 | **0.975** (1,798 false of 72,297 boxes) | same | **PASS** | `drone-test-drone.json` |
| 3 | AP50, Anti-UAV test | reported (1 and 2 imply ≥ 0.72) | **0.915**; COCO 0.402 | same | reported | `drone-test-drone.json`, `drone-test-coco.json` |
| 4 | ID kept, end to end, Anti-UAV test | ≥ 0.95 × labeled-box row (≥ 78.48%, 62,570 frames) | **75.3%** (60,042) against 82.6% (65,863) on the labels: **0.911 ×** | 79,724 frames, 91 clips | **FAIL** | `drone-test-tracking-drone-at-0.352.json` |
| 5 | Bird images with a false drone | ≤ 8 of 86 | **19 of 86** (22.1%) | 86 images, clustered | **FAIL** | `drone-test-drone-birds.json` |
| 6 | Empty-sky false alarms, Anti-UAV test | ≤ 4 detections in 2.25 min | **190** (84.3 per minute), 174 of them in one clip | 2,704 frames, 12 clips | **FAIL** | `drone-test-drone-empty-sky.json` |
| 7 | Seraphim transfer | AP50 > COCO's 0.216 | **0.164** | 8,349 images, 8,627 drones | **FAIL** | `drone-test-drone-seraphim.json` |
| 8 | Recall on tiny and small drones, Anti-UAV test | reported | tiny **0 of 39**, small **397 of 823** (0.482) | 862 boxes | reported | `drone-test-drone.json` |
| 9 | Speed, shipped configuration | 120+ FPS, P99 < 16 ms | 195.59–198.63 FPS, P99 6.46–6.49 ms (#46, a val clip) | 1 clip, 2 rounds | PASS (#46) | `drone-speed.md` |

## Anti-UAV test: the detector

| Model | Threshold | Chosen by | Recall | Precision | F1 | False alarms on every drone-absent frame (4.71 min) |
|---|---|---|---|---|---|---|
| **drone** | **0.352** | best F1 on val, frozen | **0.884** (70,499 of 79,724) | **0.975** | **0.927** | 415 (88.1 / min) |
| drone | 0.25 | the tracker's default | 0.891 (71,079) | 0.968 | 0.928 | 458 (97.2 / min) |
| COCO | 0.066 | best F1 on val, frozen | 0.457 (36,458) | 0.448 | 0.452 | 846 (179.6 / min) |
| COCO | 0.25 | the tracker's default | 0.222 (17,771) | 0.757 | 0.344 | 36 (7.6 / min) |

AP50 (scores ≥ 0.01, TensorRT): drone **0.915**, COCO **0.402**. On val they were 0.995 and 0.228 (`drone-model.md`).

Recall by labeled box size (area at 1920×1080: tiny < 16², small < 32², medium < 96², large ≥ 96² px):

| Size | Drones | Drone at 0.352 | Drone at 0.25 | COCO at 0.25 |
|---|---|---|---|---|
| tiny | 39 | 0 of 39 | 0 of 39 | 0 of 39 |
| small | 823 | 0.482 (397) | 0.522 | 0.512 |
| medium | 45,135 | 0.878 | 0.886 | 0.242 |
| large | 33,727 | 0.903 | 0.907 | 0.190 |

Recall by test session at 0.352 (`drone-test-drone-breakdown.json`; COCO's at 0.25 is in `drone-test-coco-breakdown.json`):

| Session | Drone frames | Drone recall at 0.352 | COCO recall at 0.25 |
|---|---|---|---|
| 20190925_111757 | 8,853 | **0.689** | 0.023 |
| 20190925_124000 | 9,164 | **0.670** | 0.004 |
| 20190925_124612 | 7,164 | 0.856 | 0.003 |
| 20190925_134301 | 6,802 | 0.872 | 0.108 |
| 20190925_193610 | 8,333 | 0.992 | 0.537 |
| 20190925_200805 | 7,488 | 0.860 | 0.453 |
| 20190926_095902 | 7,953 | 0.982 | 0.378 |
| 20190926_102042 | 7,519 | 0.991 | 0.199 |
| 20190926_111509 | 8,027 | 0.995 | 0.312 |
| 20190926_134054 | 8,421 | 0.974 | 0.221 |

- **Recall fell from val's 0.992 to 0.884 on test**, still above the 0.80 bar. By session it ranges from 0.670 to 0.995, against val's 0.985–0.996. Five sessions are at 0.974 or more. The other five are what moved it.
- **The two weakest sessions miss the drone outright, not narrowly.** On three of their clips, 1,604 of 2,986 drone frames have no box at 0.352, and 130 have a box at IoU under 0.5 (`drone-test-inspection.json`). The middle missed frame of `20190925_124000_1_5` (frame 441) has no detection at all at the 0.01 floor. Looked at by eye, that frame and frame 581 of `20190925_111757_1_2` show a large drone (116 × 81 and 156 × 81 px) in front of buildings, not sky. Two frames were looked at, so this says what those misses look like, not that buildings cause them.
- **COCO also does better on test than on val** (AP50 0.402 against 0.228, recall at 0.25 0.222 against 0.142). Test is a different mix of sessions, not only harder.

## Anti-UAV test: end-to-end tracking

ID kept on every frame where the drone is labeled, with misses counted. The same clips' labeled boxes go through the same tracker in the same run (`tether-antiuav --detections`).

| Detector | Tracker cutoff | ID kept | Frames kept | File |
|---|---|---|---|---|
| **drone** | **0.352** | **75.3%** | 60,042 of 79,724 | `drone-test-tracking-drone-at-0.352.json` |
| COCO | 0.25 | 15.4% | 12,318 | `drone-test-tracking-coco-at-0.25.json` |
| COCO | 0.066 | 14.1% | 11,250 | `drone-test-tracking-coco-at-0.066.json` |
| **labeled boxes** (reference) | any | **82.6%** | 65,863 | the `labeled_boxes` row of each file |

The labeled-box row is 65,863 frames in all three files. It's the same 82.6% as #28's final run on these clips (`antiuav-dropout.md`). For reference, the [reachable ceiling](../CONCEPTS.md#reachable-ceiling) on the labels at the 30-frame search is 94.2% on test (`antiuav-ceiling.md`).

**Target 4 fails: 0.911 × the labeled-box row, against ≥ 0.95.** Passing needed 62,570 frames. The drone model kept 60,042, 2,528 short.

Per session, frames kept (from the per-clip numbers in the same file):

| Session | Drone frames | Drone model | Labeled boxes | Drone − labels (frames) | COCO at 0.25 |
|---|---|---|---|---|---|
| 20190925_111757 | 8,853 | 0.611 | 0.904 | −2,591 | 0.017 |
| 20190925_124000 | 9,164 | 0.519 | 0.929 | −3,758 | 0.001 |
| 20190925_124612 | 7,164 | 0.693 | 0.801 | −768 | 0.001 |
| 20190925_134301 | 6,802 | 0.726 | 0.818 | −626 | 0.086 |
| 20190925_193610 | 8,333 | 0.779 | 0.742 | +308 | 0.322 |
| 20190925_200805 | 7,488 | 0.629 | 0.408 | +1,660 | 0.154 |
| 20190926_095902 | 7,953 | 0.787 | 0.807 | −157 | 0.326 |
| 20190926_102042 | 7,519 | 0.895 | 0.880 | +112 | 0.124 |
| 20190926_111509 | 8,027 | 0.954 | 0.965 | −88 | 0.293 |
| 20190926_134054 | 8,421 | 0.960 | 0.949 | +87 | 0.215 |

- **The shortfall follows the detector's misses.** The two sessions with the lowest recall, 0.689 and 0.670, lose 6,349 frames against the labels, more than the whole net shortfall of 5,821. The worst clips are all in `20190925_124000`: on `_1_5` the drone model keeps 119 frames against the labels' 927, and on `_1_7` 144 against 940. These are runs of missed frames, which the [dropout test's](antiuav-dropout.md) random deletions never made: the case ADR 0004 said target 4 was there to catch.
- **The drone model beats the labels on 28 of 91 clips** (3,366 frames in all). It trails on 50 (9,187 frames), and 13 are equal. Most of the gain is in two evening sessions. In `20190925_200805` the labels keep 0.408 of frames and the drone model 0.629. Clip by clip, the drone model keeps 903 frames against 167 on `20190925_200805_1_3`, 628 against 219 on `_1_5`, and 883 against 344 on `20190925_193610_1_1`. Why the labeled boxes do so badly on these clips wasn't looked into. So the val explanation (the labels mark a half-visible drone absent, or the box height jumps) isn't confirmed here.
- **COCO's tracking is bounded by its detections**, as on val: 15.4% at 0.25 and 14.1% at 0.066.

## Empty sky and drone-absent frames

| Model | Threshold | Empty-sky alarms (2.25 min) | Per minute | Every drone-absent frame (4.71 min) | File |
|---|---|---|---|---|---|
| **drone** | **0.352** | **190** | **84.3** | 415 | `drone-test-drone-empty-sky.json` |
| drone | 0.25 | 198 | 87.8 | 458 | same |
| COCO | 0.066 | 437 | 193.9 | 846 | `drone-test-coco-empty-sky.json` |
| COCO | 0.25 | 8 | 3.5 | 36 | same |

**Target 6 fails: 190 detections against ≤ 4.** By clip, counted from the dump at 0.352 on the listed frames (`drone-test-inspection.json`), the 190 are:
- **174 in `20190925_200805_1_7`, a night clip.** Looked at by eye on frames 184, 550 and 760, it's one lamp on a lit structure: a red and a white point light. The box stays on it (about 47 × 40 px) as the camera tilts. A drone at night in this session is also a lit point. The COCO model fires at the same spot (7 of its 8 alarms at 0.25), so the lamp is a hard negative, not a labeling mistake.
- **13 in `20190926_095902_1_3`,** on a structure at the bottom-left edge of the frame (frame 280 looked at).
- **1 each** in three other clips.

So the alarms are recurring false objects, the kind the bar was built to fail, and they fail it: 174 of the 190 (91.6%) are one lamp.

## Birds

| Model | Threshold | Images with any detection | Rate | File |
|---|---|---|---|---|
| **drone** | **0.352** | **19 of 86** | **22.1%** | `drone-test-drone-birds.json` |
| drone | 0.25 | 21 of 86 | 24.4% | same |
| COCO | 0.25 | 83 of 86 | 96.5% | `drone-transfer-coco-birds.json` (#45) |

**Target 5 fails: 19 images against ≤ 8.** Each alarmed image has one box. Looked at by eye as a contact sheet (not committed: the images are third-party):
- **8 are a whole small bird in flight against the sky,** from two dusk scenes behind a minaret and a dome (`BT (211)`–`BT (230)`). That's close to the operational case the ADR said this row couldn't test: a distant bird against sky, called a drone.
- **11 are small boxes on part of a close-up bird or the background:** a beak, feet, a branch.

The images sit in a few scenes, so the 19 aren't 19 independent errors. The 8 sky cases are two scenes.

## Seraphim test (transfer)

| Model | Threshold | AP50 | Recall | Precision | F1 | File |
|---|---|---|---|---|---|---|
| **drone** | **0.352** (frozen on Anti-UAV) | **0.164** | 0.180 (1,559 of 8,627) | 0.543 | 0.271 | `drone-test-drone-seraphim.json` |
| drone | 0.25 | | 0.193 (1,673) | 0.491 | 0.278 | same |
| COCO | 0.066 | **0.216** | 0.608 (5,248) | 0.193 | 0.293 | `drone-transfer-coco-seraphim.json` (#45) |
| COCO | 0.25 | | 0.372 (3,212) | 0.333 | 0.352 | same |

**Target 7 fails: AP50 0.164, under the COCO model's 0.216 on the same images, runtime and floor.** AP50 doesn't depend on the threshold, so the frozen 0.352 isn't the reason.

Recall by box size (area at Seraphim's 640×640):

| Size | Drones | Drone at 0.352 | Drone at 0.25 | COCO at 0.25 |
|---|---|---|---|---|
| tiny | 626 | 0.004 (3) | 0.007 | 0.315 |
| small | 1,425 | 0.256 | 0.291 | 0.474 |
| medium | 2,741 | 0.389 | 0.409 | 0.377 |
| large | 3,835 | **0.032** (124) | 0.033 | 0.340 |

- **The drone model only matches COCO on medium boxes. It loses almost all the large ones**, which are 44% of Seraphim's drones. On the 3,723 images with a large drone, the median top score is 0.027, and 3,049 have no box at 0.352 (`drone-test-inspection.json`). Fifteen of those, drawn at random (seed 0), were looked at, and all 15 are close-ups: drones held in hand, product photos on white, top-down studio shots, a rotated frame with salt-and-pepper noise. Fifteen images show what these misses look like, not how many of the 3,049 are close-ups. Anti-UAV has none of these. Its large boxes are a drone in the sky tens of pixels across.
- **It also loses COCO's tiny and small hits** (0.004 against 0.315 tiny). Seraphim's tiny boxes are measured at 640×640, so they don't line up with Anti-UAV's.
- That's consistent with the fine-tune narrowing the model to one site's look, a drone some distance away against sky. It's also consistent with the COCO model's "airplane" and "kite" hits on Seraphim ([drone-transfer-sets](drone-transfer-sets.md)) having been lost with COCO's classes. This run doesn't separate the two. ADR 0004 names the next experiment if this target fails: training on Seraphim.

## What it means

- **On new flights at the training site, the drone model finds the drone on 88.4% of frames at 97.5% precision**, against the COCO model's 22.2% at 75.7% at the tracker's 0.25. Those two pass their bars with room, and AP50 goes from 0.402 to 0.915.
- **It doesn't yet track as well as the labels do.** Its misses come in long runs on two sessions, and there it costs the tracker 6,349 frames. The bar was set to catch exactly that.
- **It isn't a general drone detector.** It's worse than the untrained COCO model on Seraphim. On this site it fires all night on a lamp, and it calls a bird in flight against the sky a drone in two scenes.
- **The claim stays narrow**, narrower than the ADR hoped. "The perception part of a counter-UAS pipeline, measured on one site's footage" holds for detection recall and precision on that site. ID retention, the false-alarm rate and transfer all miss their bars.

## Caveats (they travel with every number above)

- **Every number names its model.** Rows are `yolo26n_drone_1280x736` (the drone model, epoch 19) or `yolo26n_1280x736` (COCO, class-agnostic).
- **Two recording days, one camera.** All of Anti-UAV, test included, is one pan-tilt turret on 2019-09-25 and 26. Test's 10 sessions are new flights, but the same site and camera. Its recall spread across sessions (0.670–0.995) shows how much a session can move a number.
- **Seraphim mixes real photos, marketing images and synthetic renders,** merged from 23 sources. It's padded to 640×640, so a 1280×736 input upscales it about 1.15×. Its size buckets are at 640×640, not Anti-UAV's 1920×1080. All 15 of its missed large-drone images looked at are close-ups no counter-UAS camera would see, and the large boxes are where the drone model falls furthest behind COCO.
- **The bird images are mostly close-up web photos, clustered in a few scenes.** One image is 1.2 points of rate. 8 of the 19 alarms are two sky scenes.
- **Empty sky is 2.25 minutes from one site,** and 174 of its 190 alarms are one lamp in one night clip. It counts detections, not objects: one object lasting 174 frames is 174 alarms.
- **The threshold and tracker cutoff were frozen on Anti-UAV val.** They may not suit Seraphim, but AP50 doesn't depend on them.
- **The labeled-box row is a reference, not a ceiling.** The drone model beats it on 28 clips, and the reason on test wasn't looked into.
- **What was looked at by eye:** 2 missed Anti-UAV frames, 4 empty-sky frames, the 19 alarmed bird images and 15 missed Seraphim images. They describe what the errors look like, not how often each kind occurs.
- **Speed is #46's:** one laptop (RTX 5070 Laptop GPU), one val clip, two rounds, decode-bound.
- **One run.** One training run (seed 0), scored once. Run-to-run variation of training wasn't measured.

## Reproduce

From the main checkout, with the data in `data/` and the engines built beside the models. `DET` is the detector command used here: `"<build>/Release/tether-pipeline.exe" --runtime tensorrt --prep cuda`, quoted so `cmd` accepts the path.

```
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_drone_1280x736.onnx --split test --det-min-score 0.01 \
    --work-dir results/antiuav-trt --detector "$DET"
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_1280x736.onnx --split test --det-min-score 0.01 \
    --work-dir results/antiuav-trt --detector "$DET"
tether-image-detect data/seraphim/test/images --list splits/seraphim-test.json --set seraphim-test \
    --model models/yolo26n_drone_1280x736.onnx --det-min-score 0.01 --work-dir results/drone-transfer --detector "$DET"
tether-image-detect "data/DroneVsBird/<...>/Dataset/test/images" --list splits/drone-vs-bird-bt.json --set drone-vs-bird-bt \
    --model models/yolo26n_drone_1280x736.onnx --det-min-score 0.01 --work-dir results/drone-transfer --detector "$DET"

tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_drone_1280x736/test --split test \
    --threshold 0.3524462878704071 --out docs/benchmarks/drone-test-drone.json
tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_drone_1280x736/test --split test \
    --threshold 0.3524462878704071 --empty-sky splits/antiuav-empty-sky.json --out docs/benchmarks/drone-test-drone-empty-sky.json
tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_1280x736/test --split test \
    --threshold 0.06577478349208832 --out docs/benchmarks/drone-test-coco.json        # and --empty-sky ... drone-test-coco-empty-sky.json
python tools/drone_breakdown.py data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_drone_1280x736/test --split test \
    --thresholds 0.3524462878704071 0.25 --out docs/benchmarks/drone-test-drone-breakdown.json   # COCO: 0.0658 0.25
tether-drone-score-images results/drone-transfer/detections/yolo26n_drone_1280x736/seraphim-test \
    --labels data/seraphim/test/labels --threshold 0.3524462878704071 --out docs/benchmarks/drone-test-drone-seraphim.json
tether-drone-score-images results/drone-transfer/detections/yolo26n_drone_1280x736/drone-vs-bird-bt --drone-free \
    --threshold 0.3524462878704071 --out docs/benchmarks/drone-test-drone-birds.json
tether-antiuav data/Anti-UAV300 --detections results/antiuav-trt/detections/yolo26n_drone_1280x736/test \
    --work-dir results/antiuav-e2e-test/drone-at-0.352 --name drone-at-0.352 \
    --tracker '"<build>/Release/tether-track.exe" --min-score 0.3524462878704071'
```

The COCO tracking rows change the dump folder to `yolo26n_1280x736/test` and `--min-score` to 0.25 and 0.06577478349208832. `tether-antiuav` writes `results.json` under its work dir, copied here as `drone-test-tracking-<row>.json`. The tools were run with `--split-file` pointing at this branch's `splits/antiuav.json`, which is what the result files record. The drone model's Anti-UAV test dump took 11 min 22 s, COCO's 11 min 4 s, Seraphim 1 min 55 s and the birds 7 s.
