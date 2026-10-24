# Drone detection: the transfer and false-alarm test sets, with the COCO "before" rows

Built 2026-10-22 for #45, following [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md). These are the sets the drone model is scored on once, in #48, after #47 sets its targets:
- **Seraphim test**, the transfer test, with Anti-UAV frames looked for and dropped.
- **Drone-vs-Bird bird test images**, near-duplicates removed, for the per-image bird false-alarm rate.
- **Empty sky**: the Anti-UAV frames that count as empty sky, by a rule fixed on val. The rule replaces "every frame with the drone labeled absent".

The only detector run on the two outside sets is the pretrained COCO **YOLO26n at 1280×736** (`yolo26n_1280x736`), scored class-agnostically: any detection counts as a drone. Nothing was tuned on these sets. The drone model (`yolo26n_drone_1280x736`) wasn't run on Seraphim, the birds or Anti-UAV test. Its only number here is from its existing val dump (#43). No detector was run on Anti-UAV test; its empty-sky frames were picked from labels and frame indices alone.

*Update (2026-10-23, #48): the drone model's rows on these sets are in [drone-test](drone-test.md): Seraphim AP50 0.164, 19 of 86 bird images and 190 empty-sky detections at 0.352, all missing their targets.*

Committed lists (no images; Seraphim and Drone-vs-Bird are third-party and scraped, so none go in git or `docs/media`):
- [`splits/seraphim-test.json`](../../splits/seraphim-test.json): the kept and dropped Seraphim test images, with the overlay rule and the 20 highest-scoring kept images.
- [`splits/seraphim-test-eye-checked.json`](../../splits/seraphim-test-eye-checked.json): the 432 Seraphim images looked at by eye.
- [`splits/drone-vs-bird-bt.json`](../../splits/drone-vs-bird-bt.json): the kept bird images, and each dropped one with the kept image it duplicates.
- [`splits/antiuav-empty-sky.json`](../../splits/antiuav-empty-sky.json): the rule, and the val and test frames it selects, per clip.

Raw results:
- [`drone-transfer-overlay-check.json`](drone-transfer-overlay-check.json): real Anti-UAV frames under the overlay check.
- [`drone-transfer-coco-seraphim.json`](drone-transfer-coco-seraphim.json) and [`drone-transfer-coco-birds.json`](drone-transfer-coco-birds.json): the COCO rows.
- [`drone-empty-sky-val-coco.json`](drone-empty-sky-val-coco.json) and [`drone-empty-sky-val-drone.json`](drone-empty-sky-val-drone.json): the val empty-sky rows.

## The sets

| Set | Source | Images or frames | After cleaning | Ground truth |
|---|---|---|---|---|
| Seraphim test | `lgrzybowski/seraphim-drone-detection-dataset`, CC BY 4.0 | 8,349 images, 8,627 labeled drones | **8,349 kept, 0 dropped** as Anti-UAV | its YOLO labels, one class |
| Drone-vs-Bird birds | Roboflow `drone-3doih/drone-76w4q` (YOLOv7 export), CC BY 4.0: test images whose names start `BT` | 361 images | **86 kept**, 275 near-duplicates dropped | the filename: every `BT` image is drone-free; label files are never read |
| Empty sky, val | Anti-UAV300, the session-disjoint val (`splits/antiuav.json`, seed 0) | 1,232 drone-absent frames (1.03 min) | **605 frames, 0.50 min**, in 4 clips | the rule below |
| Empty sky, test | Anti-UAV300 official test | 5,650 drone-absent frames (4.71 min) | **2,704 frames, 2.25 min**, in 12 clips | the rule below |

### Seraphim: looking for Anti-UAV frames

Every Anti-UAV visible-light frame has the turret's reticle: twelve 2-px line segments at fixed places in the 1920×1080 frame, each pixel drawn black or white to stand out from what's behind it. `tether.antiuav_overlay.reticle_score` maps an image back to 1920×1080, two ways:
- as is, and
- as a 16:9 frame letterboxed into a square, which is how Seraphim stores a wide image (640×360 inside 640×640, padded with gray 114).

A first draft looked for bright lines and scored real frames as low as −124, because the reticle is drawn black over bright sky. So along each segment, pixel by pixel, the score measures how far the line stands out from the bands on both sides, in the same direction, either dark or bright. It subtracts the most that the same measure finds 14–50 px away across the segment, so edges, texture and grids that light up everywhere cancel. The score is the median over the 12 segments. An image is dropped at a score of 14 or more.

- **Real Anti-UAV frames are all flagged.** 681 frames (3 drawn at random, seed 0, from each of the 227 train and val clips of the split; never test) were scored in five forms:

  | Form | Flagged | Lowest score | Median |
  |---|---|---|---|
  | as recorded (1920×1080) | 681 of 681 | 48.6 | 81.6 |
  | letterboxed into 640×640, JPEG quality 75 | 681 of 681 | 36.1 | 71.5 |
  | letterboxed into 416×416, JPEG quality 75 | 681 of 681 | 21.0 | 45.1 |
  | stretched to 640×640 | 681 of 681 | 39.5 | see JSON |
  | mirrored, letterboxed into 640×640 | 681 of 681 | 21.1 | see JSON |

- **No Seraphim test image is flagged.** All 8,349 score 0 or below; the 20 highest are listed in `splits/seraphim-test.json`.
- **Checked by eye: 432 images, none Anti-UAV.** 384 were drawn at random (seed 1) and 48 were the highest under a draft of the score, viewed as contact sheets. None showed the reticle, the turret's text, or the site. With 0 of 384 random images, Anti-UAV frames in any form are unlikely to be more than about 1% of the set (the rule of three: 3 / 384 = 0.8% at 95% confidence).

So **0 images were dropped**, and Seraphim test is used whole: 8,349 images.

### Drone-vs-Bird: near-duplicates

`tether.near_duplicates`: a 64-bit pHash per image (32×32 grayscale, DCT, the 8×8 lowest frequencies against their median). The `BT` images are walked in number order, and one is kept unless it's within **10 bits** of an image already kept. Ten bits is a common pHash cut for "the same picture". On these images, pairs that close were frames of one shot with the bird barely moved.

- 361 bird images, **86 kept**. 89 of the 361 have exactly the same hash as another image (272 distinct hashes).
- Most drops are frames from a few bird videos: `BT (25)` absorbs 32 images and `BT (158)` 31.
- Pairs at 14–20 bits are often still the same scene with the bird in another pose, so the 86 aren't 86 independent scenes.

### Empty sky: the rule

#43 and #44 found that most of the drone model's "false alarms" on val drone-absent frames were the drone half out of view, which the labels mark absent. So "drone labeled absent" isn't empty sky. The rule (`tether.empty_sky`), fixed before any test run:

> A frame is empty sky if the drone is labeled absent and **no labeled drone is within 2 s (40 frames at 20 fps)** before or after it in the same clip. The frames just outside the clip count as possibly holding the drone, since each clip is a piece of a longer recording.

- **Why 2 s.** It's past the furthest drone-absent detection #43 found from a labeled drone (36 frames; the next are 35, 27 and 21, in [`drone-model-absent-detections.json`](drone-model-absent-detections.json)), and past the tracker's 30-frame search. It's a round number, but only 4 frames past that furthest detection, so it was chosen with those detections in view.
- **How it was sized.** The margin was chosen on val labels. Val's 1,232 drone-absent frames fall in 14 runs, the longest 564 frames. Kept at a margin of 0.5 / 1 / 2 / 5 s: 973 / 809 / 605 / 364 frames (`val_sizing` in the list file).
- **Test.** The rule was then applied to test by labels and frame index only. No test pixels or detections were looked at.

| Split | Drone-absent frames | Empty sky | Clips | Largest clip's share |
|---|---|---|---|---|
| val | 1,232 (1.03 min) | **605 (0.50 min)** | 4 | 484 frames (80%), one 564-frame absence |
| test | 5,650 (4.71 min) | **2,704 (2.25 min)** | 12 | 690 frames (26%) |

## Results: the COCO "before" rows

Model: `yolo26n_1280x736`, the pretrained COCO weights, class-agnostic. Detections came from the C++ pipeline on TensorRT at a **0.01 floor**, the same runtime, engine (`trtexec --noTF32`) and floor as every drone-milestone row. The thresholds are **0.066**, its best-F1 cut on val (frozen from [drone-coco-baseline](drone-coco-baseline.md)), and the tracker's **0.25**.

### Seraphim test (transfer)

AP50 **0.216** (scores ≥ 0.01), against 0.228 on the Anti-UAV val at the same floor.

| Threshold | Chosen by | Recall | Precision | F1 |
|---|---|---|---|---|
| 0.066 | best F1 on Anti-UAV val | 0.608 (5,248 of 8,627) | 0.193 | 0.293 |
| 0.25 | the tracker's min score | 0.372 (3,212 of 8,627) | 0.333 | 0.352 |

Recall by box size (area at Seraphim's 640×640: tiny < 16², small < 32², medium < 96², large ≥ 96² px):

| Size | Drones | Recall at 0.066 | Recall at 0.25 |
|---|---|---|---|
| tiny | 626 | 0.580 | 0.315 |
| small | 1,425 | 0.680 | 0.474 |
| medium | 2,741 | 0.592 | 0.377 |
| large | 3,835 | 0.598 | 0.340 |

The classes it reports at 0.25: person 3,563 detections, airplane 2,449, car 540, kite 340, cell phone 293, bird 268. The by-class counts were made from the detection file for this write-up and aren't in the results JSON.

### Drone-vs-Bird birds (per-image false alarms)

| Threshold | Images with any detection | Rate |
|---|---|---|
| 0.066 | 86 of 86 | 100% |
| 0.25 | 83 of 86 | 96.5% |

On 81 of the 86 images the top-scored box at 0.066 is COCO's "bird". This row is the metric's upper end, not a finding: scored class-agnostically, a model that knows birds will report them. It shows the bird row isn't trivially 0 for a model that sees birds.

### Empty sky on val

The existing val dumps (#43), rescored with `--empty-sky`. No new detector run:

| Model | Threshold | Alarms on empty sky (0.50 min) | Per minute | Alarms on every drone-absent frame (1.03 min) |
|---|---|---|---|---|
| COCO | 0.066 | 9 | 17.9 | 141 |
| COCO | 0.25 | 0 | 0 | 8 |
| drone, epoch 19 | 0.352 | **0** | **0** | 48 |
| drone, epoch 19 | 0.25 | 3 | 6.0 | 64 |

## What it means

- **The rule removes what it was meant to.** The drone model's 48 val "false alarms" at 0.352, nearly all a half-visible drone, drop to 0. That's the expected direction, but the margin was justified partly by those same 48 detections, so it isn't independent evidence. Test is.
- **Seraphim is a different kind of picture.** The COCO model finds 37.2% of Seraphim drones at 0.25, against 14.2% of Anti-UAV val drones. Many Seraphim images are close-ups, product shots and hand-held drones that COCO calls "airplane", "kite" or "bird". Its precision is low because the images also hold people, cars and phones, which are false positives when every class counts.
- **The bird row needs a drone-only model to mean anything.** For COCO it's 96.5% at 0.25 because COCO has a bird class. For the drone model it measures whether birds look like drones to it.

## Caveats (they travel with every number from these sets)

- **Every number names its model.** These rows are `yolo26n_1280x736` (COCO), except the val drone-model row, which is `yolo26n_drone_1280x736_e19`. #48 adds the drone model's test rows.
- **Seraphim mixes real photos, marketing images and synthetic renders** (its README). Its labels were merged from 23 sources without relabeling.
- **Seraphim is padded to 640×640**, so a 1280×736 input upscales it about 1.15× (736 / 640). Its size buckets are at 640×640, not at Anti-UAV's 1920×1080, so they don't line up with Anti-UAV's.
- **The overlay check has blind spots.** It catches an Anti-UAV frame resized, letterboxed, stretched, mirrored or JPEG-compressed. It doesn't catch one rotated, sheared or cropped, which Roboflow augmentations do; some Seraphim images are visibly rotated with salt-and-pepper noise. It also doesn't know Anti-UAV's infrared overlay, which isn't in the local copy. The eye check (0 of 384 random images) bounds what it could miss to about 1% of the set, not 0.
- **Close-up birds and empty sky are the easy cases.** Drone-vs-Bird's bird images are mostly close-up web photos, and empty sky is just that. A low rate on either doesn't show the model can tell a distant bird from a distant drone.
- **The bird set is small and clustered.** 86 images, many from the same few videos. One image is 1.2 points of rate.
- **Empty sky is thin.** Val's 0.50 min is 80% one clip, so a val rate is a handful of detections. Test's 2.25 min (12 clips) is better but still a few minutes from one site. It counts detections, not frames.
- **Image decoding.** The pipeline reads the images as a numbered sequence through OpenCV's FFmpeg backend. On 20 Seraphim images, that decode differed from `cv2.imread` by a mean of 0–1.4 gray levels per image. The model, letterbox and CUDA prep are the pipeline's own, the same as for the Anti-UAV clips.

## How it's scored

`tether-drone-score-images` (`python/tether/drone_score.py`), the scorer's image-folder mode:
- **Seraphim.** YOLO-format labels, any number of drones per image. Detections claim drones in score order, each taking the unclaimed drone it overlaps most at IoU ≥ 0.5. Every other detection is a false positive. With one drone per image this is the clip scorer's rule. AP50 and the rows are computed the same way as on Anti-UAV.
- **Birds.** `--drone-free`: the share of images with any detection at or above the threshold. An image counts once however many boxes it has.
- **Empty sky.** `tether-drone-score --empty-sky splits/antiuav-empty-sky.json` counts false alarms only on the listed frames, over their duration. A listed frame with the drone labeled is refused.
- **Tested.**
  - The null cases: nothing, the same box on every image, and a whole-image box score AP50 ≈ 0; copying the labels scores 1.
  - The bird rate: no detections give 0%, a box on every image gives 100% (`python/tests/test_drone_score_images.py`).
  - The overlay check on synthetic frames (`test_antiuav_overlay.py`). On real frames it's checked by the `overlay-check` run and the Seraphim scan above, since the datasets aren't in git. The committed lists were rebuilt from the data and came out identical.
  - Dedup (`test_near_duplicates.py`) and the set builders (`test_transfer_sets.py`).
  - The empty-sky rule and its committed list (`test_empty_sky.py`), and the image dump (`test_image_detect.py`).
  - Mutation check: [`docs/spec-verify/issue-45.md`](../spec-verify/issue-45.md).

## Reproduce

Seraphim's test images and labels come as `test/images/batch_001.zip` and `test/labels/batch_001.zip`. Unzip each into its own folder first. Then, from the repo root, with data in `data/`:

```
tether-transfer-sets seraphim data/seraphim/test/images                  # about 3 minutes
tether-transfer-sets birds "data/DroneVsBird/<...>/Dataset/test/images"
tether-transfer-sets overlay-check data/Anti-UAV300                       # train and val frames only
tether-empty-sky data/Anti-UAV300
tether-image-detect data/seraphim/test/images --list splits/seraphim-test.json --set seraphim-test \
    --model models/yolo26n_1280x736.onnx --det-min-score 0.01 --work-dir results/drone-transfer \
    --detector "\"cpp/build-trt/Release/tether-pipeline.exe\" --runtime tensorrt --prep cuda"
tether-drone-score-images results/drone-transfer/detections/yolo26n_1280x736/seraphim-test \
    --labels data/seraphim/test/labels --threshold 0.066 --out docs/benchmarks/drone-transfer-coco-seraphim.json
tether-image-detect "data/DroneVsBird/<...>/Dataset/test/images" --list splits/drone-vs-bird-bt.json \
    --set drone-vs-bird-bt --model models/yolo26n_1280x736.onnx --det-min-score 0.01 \
    --work-dir results/drone-transfer --detector "\"cpp/build-trt/Release/tether-pipeline.exe\" --runtime tensorrt --prep cuda"
tether-drone-score-images results/drone-transfer/detections/yolo26n_1280x736/drone-vs-bird-bt --drone-free \
    --threshold 0.066 --out docs/benchmarks/drone-transfer-coco-birds.json
tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_1280x736/val --split val \
    --threshold 0.06577478349208832 --empty-sky splits/antiuav-empty-sky.json --out docs/benchmarks/drone-empty-sky-val-coco.json
```

The drone model's val row is the same last command on `yolo26n_drone_1280x736_e19/val` at `--threshold 0.3524462878704071`. On Windows the detector path is quoted so `cmd` accepts its forward slashes. The committed rows were run from the main checkout against this branch's split files, so their JSON records those paths. The Seraphim dump took 1 min 49 s for 8,349 images.

For #48, the drone model's rows are the same commands with `--model models/yolo26n_drone_1280x736.onnx` and `--threshold 0.3524462878704071`, and `--empty-sky` on the test dump with `--split test`.
