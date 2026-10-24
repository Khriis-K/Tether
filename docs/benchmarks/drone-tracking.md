# Drone tracking end to end: the tracker on a real detector's output

Run 2026-10-22 for #44. Until now every Anti-UAV tracking number gave the tracker the *labeled* boxes, with some deleted on purpose ([antiuav-dropout](antiuav-dropout.md)). Here the tracker runs on what a detector actually found, on the **session-disjoint val** (`splits/antiuav.json`, seed 0: 42 clips, 38,250 frames with the drone labeled), scored with the same ID-kept rule, beside the same clips run on their labeled boxes ([ADR 0004](../adr/0004-drone-detector-on-anti-uav.md)). The detector's own misses are the dropout, so nothing is deleted on top. Anti-UAV test and Seraphim weren't touched.

*Update (2026-10-23, #48): on Anti-UAV test, run once with the C++ tracker at 0.352, the drone model keeps the ID on 75.3% of frames against 82.6% on the labeled boxes (0.911 ×, missing the ≥ 0.95 × target). The COCO model keeps 15.4% at 0.25. See [drone-test](drone-test.md).*

Two detectors, both YOLO26n at 1280×736 on TensorRT with scores saved down to 0.01 (the dumps #43 scored):
- **`yolo26n_drone_1280x736_e19`**: the fine-tuned drone model ([drone-model](drone-model.md)). The shipped file `models/yolo26n_drone_1280x736.onnx` is byte-identical to it (same SHA-256), so its val dump was reused.
- **`yolo26n_1280x736`**: the pretrained COCO model, every class handed over as a drone (the "before" row, [drone-coco-baseline](drone-coco-baseline.md)).

The tracker is the Python `tether-track` with its defaults except the score cutoff (`--min-score`) and, in the ablations, `--low-score-ratio`. Raw results, one file per row, each with every clip's numbers: `drone-tracking-<row>.json` beside this file, from `tether-antiuav --detections`.

## Results

ID kept: of the 38,250 val frames with the drone labeled, the share where the clip's most common ID holds the drone (a "tracked" track with IoU ≥ 0.5 against the labeled box), summed over clips. A frame the detector missed counts against it.

| Detector | Tracker cutoff | Low band | ID kept | Frames kept | Results file |
|---|---|---|---|---|---|
| **drone model** | **0.352** (its best-F1 threshold) | 0.141–0.352 (the rule) | **89.7%** | 34,317 | `drone-tracking-drone-at-0.352.json` |
| drone model | 0.25 (the tracker's default) | 0.1–0.25 (the rule) | 89.7% | 34,325 | `drone-tracking-drone-at-0.25.json` |
| COCO, class-agnostic | 0.25 (the tracker's default) | 0.1–0.25 (the rule) | 10.9% | 4,181 | `drone-tracking-coco-at-0.25.json` |
| COCO, class-agnostic | 0.066 (its best-F1 threshold) | 0.026–0.066 (the rule) | 13.2% | 5,043 | `drone-tracking-coco-at-0.066.json` |
| **labeled boxes** (no detector) | any | none (every box scores 1) | **87.0%** | 33,288 | the `labeled_boxes` row of each file |

The labeled-box row comes out identical in every file: 33,288 frames. It's the ADR's "labeled-box ceiling". For reference, the best any tracker with a 30-frame search could score on the labels alone, the [reachable ceiling](../CONCEPTS.md#reachable-ceiling), is 95.1% (36,375) on these clips.

Per session, from the per-clip numbers in the same files (frames kept, of the drone-labeled frames):

| Session | Drone frames | Drone model at 0.352 | Labeled boxes | COCO at 0.25 |
|---|---|---|---|---|
| 20190925_131530 | 6,889 | 0.949 | 0.917 | 0.005 |
| 20190925_133630 | 9,443 | 0.864 | 0.839 | 0.044 |
| 20190925_205804 | 6,927 | 0.763 | 0.762 | 0.022 |
| 20190926_142435 | 8,036 | 0.932 | 0.914 | 0.144 |
| 20190926_143632 | 6,955 | 0.985 | 0.925 | 0.349 |

## The C++ pipeline gives the same result (#46)

*Added 2026-10-23 for #46.* The rows above ran the Python tracker. The C++ path was then run the same way: the drone model's engine rebuilt from the shipped ONNX file, a fresh val dump from `tether-pipeline --runtime tensorrt --prep cuda --dump` (the C++ detector with CUDA prep), and the C++ `tether-track --min-score 0.3524462878704071` with the band rule's default ratio. `tether-pipeline` doesn't write track files, so dumping and tracking are its two halves run apart, on the same code.

| Dump | Tracker | ID kept | Frames kept | Results file |
|---|---|---|---|---|
| #43's (`yolo26n_drone_1280x736_e19`) | Python (the row above) | 89.7% | 34,317 | `drone-tracking-drone-at-0.352.json` |
| #43's | **C++** | 89.7% | 34,317 | (not saved: it matched the row above clip by clip) |
| rebuilt engine (`yolo26n_drone_1280x736`) | **C++** | **89.7%** | **34,317** | `drone-tracking-drone-cpp-at-0.352.json` |
| rebuilt engine | Python | 89.7% | 34,317 | (not saved: same as the C++ row) |

All four agree on every one of the 42 clips' counts, and the labeled-box row is 33,288 in each. On the rebuilt engine's dump, the C++ and Python track files have the same IDs, statuses and classes on all 78,964 frames (the 42 clips run on the detector and on the labeled boxes), and their boxes are exactly equal.

The rebuilt engine's dump isn't byte-identical to #43's: a TensorRT build times kernels and keeps the fastest, so two builds can add numbers in a different order. Within #16's agreement rule (score ≥ 0.25, IoU ≥ 0.95, scores at most 0.02 apart) all 76,606 checked detections have a partner, the worst score difference is 0.000064, and 17,860 of 39,482 frames are identical to the byte. Scored by `tether-drone-score`, it gives #43's numbers: AP50 0.995, and at 0.352 the same 37,961 of 38,250 drones found (`drone-model-rebuilt-engine.json`).

## The low-score band

The tracker's second matching pass lets a track that was tracked last frame keep going on a weak detection: one scored above the band's floor but under the cutoff. It never starts a track. The floor was a fixed 0.1, ByteTrack's value at its 0.25 cutoff. A model run at another threshold would get either no band at all (a cutoff of 0.1 or less) or a much wider one.

**The rule: the floor is 0.4 × the cutoff** (`LOW_SCORE_RATIO`, `--low-score-ratio` in both the Python and C++ `tether-track`). At the default 0.25 that's exactly ByteTrack's 0.1, so every earlier result is unchanged. At the drone model's 0.352 it's 0.141; at COCO's 0.066 it's 0.026. A ratio of 1 turns the pass off. The ratio is ByteTrack's own proportion (0.1 / 0.25), kept rather than tuned: it isn't fitted to this val.

| Detector, cutoff | Low band | Detections in the band (whole val) | ID kept | Frames kept |
|---|---|---|---|---|
| drone model, 0.352 | the rule: 0.141–0.352 | 373 | 89.7% | 34,317 |
| drone model, 0.352 | fixed floor: 0.1–0.352 | 477 | 89.7% | 34,321 |
| drone model, 0.352 | off | 0 | 88.5% | 33,835 |
| COCO, 0.066 | the rule: 0.026–0.066 | 59,246 | 13.2% | 5,043 |
| COCO, 0.066 | fixed floor 0.1: empty, so off | 0 | 9.5% | 3,620 |

- **For the drone model the band is worth 1.26 points (482 frames), and where its floor sits doesn't matter.** Its scores are nearly all high: 38,148 of 39,859 saved detections clear 0.352. The rule's band and the old fixed one differ by 104 detections and 4 frames kept, which is noise.
- **For the COCO model at its best-F1 cut a fixed 0.1 floor silently turns the pass off**, since the floor is above the cutoff. The rule keeps it, and that's worth 3.7 points (1,423 frames), a 39% relative gain on a weak baseline. This is the failure the issue warned about, measured.
- The fixed-floor row for the drone model was run with `--low-score-ratio 0.28373430426985226`, a floor of 0.1000011; no detection in the dump scores between 0.1 and that, so it's the same as 0.1 exactly.

## Choosing the tracker's settings on val

Only the cutoff was chosen; every other tracker setting is the default, tuned earlier on the labeled-box val ([antiuav-search-gate-cap](antiuav-search-gate-cap.md), [antiuav-duplicates](antiuav-duplicates.md)). 0.352 and 0.25 tie (34,317 against 34,325 frames, 0.02 points), so the drone model runs at **0.352**: one threshold for the detector's score and the tracker's cutoff, the one already frozen for test. The band follows by the rule.

## What it means

- **The drone model tracks the drone end to end on 89.7% of val frames**, against 10.9% for the COCO model at the same default cutoff.
- **The drone model's row is above the labeled-box row (89.7% against 87.0%), so on this val the labeled boxes aren't a ceiling.** Clip by clip, the detector costs a little almost everywhere: on 38 of the 42 clips it's 202 frames behind the labels in all (24 clips behind, 10 ahead, 4 equal). Four clips gain 1,231 frames, for two reasons that both come from the labels, not the tracker:
  - **The labels give up on a drone the detector still sees** (3 clips). On `20190926_143632_1_5` the drone is labeled absent for 40 frames (535–574). The detector finds it on the first 8 of them (scores 0.43–0.80), so the track's 30-frame search outlasts the rest. The detector run kept 952 frames; the labels allow at most 535 for any tracker with a 30-frame search. On `20190925_131530_1_7` (34 absent frames; detections on 6) it kept 861 against at most 634. On `20190926_142435_1_8`, after a 27-frame absence (473–499), a detection on frame 499, the last labeled-absent frame, re-matched the searching track; with the labels, the drone came back on frame 500 under a new ID that was never merged back (why wasn't looked into). Kept: 659 against 498. These are the same half-out-of-view frames that made the drone model's "false alarm" rate high in #43.
  - **The labeled box's height jumps during a camera snap** (1 clip). On `20190925_133630_1_1`, frames 547–550, the labeled height goes 82 → 73 → 64 → 53 px while the drone's box moves about 146 px a frame. The detector's boxes stay 62–65 px tall. With the labels, the new track confirmed at frame 550 is 1.55× shorter than the old track's last box, past the merge's 1.5× [same-size check](../CONCEPTS.md#duplicate-track-merge), so it keeps a new ID; with the detector it merges back into ID 1. Kept: 913 against 504.
- **So the labeled-box row isn't a strict upper bound for a good detector.** It assumes the labels are what a perfect detector would output, and on Anti-UAV they aren't quite: they mark a partly visible drone absent, and their box size jitters. It's still the right reference for "what a detector costs", and on most clips it's above the detector.
- **The COCO model's tracking is bounded by its detections.** It finds 14.2% of drones at 0.25 ([drone-coco-baseline](drone-coco-baseline.md)) and keeps the ID on 10.9% of frames.

## Caveats

- **Val chose everything here.** The detector's epoch and threshold (#43) and the tracker's cutoff were all chosen on this val, and the tracker's other settings on the old official val. The number for any claim is test, run once in #48.
- **Five sessions, one camera, two days.** By session the drone model keeps the ID on 76.3%–98.5% of frames, and the labeled boxes on 76.2%–92.5%, so the session spread is mostly the footage, not the detector. Gaps under about a point between rows weren't checked for noise.
- **Two of the explanations above were checked by looking at track files frame by frame, on four clips.** Whether more clips gain or lose for the same reasons wasn't counted.
- **False-alarm-style numbers don't appear here** (ID kept is scored only where the drone is labeled). But the frames where the drone model "beats" the labels are the same drone-absent frames that inflated its false-alarm rate in #43, so don't read either number without the other. Val has only 1.03 minutes of drone-absent frames.
- **Python tracker only.** The C++ `tether-track` has the same rule and option, tested with the same cases (`cpp/tests/test_low_score_band.cpp`); it wasn't run on this val.
- **Class-agnostic tracking is kind to COCO in one way and hard in another**, as in its detection score: any class is a drone, so a drone seen as "airplane" then "bird" keeps one track, but two boxes of different classes on the drone become two tracks.

## How it's scored

`tether-antiuav --detections DIR` (`evaluate_detections` in `python/tether/antiuav.py`):
- Reads the folder `tether-antiuav-detect` wrote and its `dump.json`, which names the model, score floor and split. The clips are that split's, from the split file. A folder without `dump.json`, a file of the wrong length, or a `--split` other than the dump's is refused.
- Hands every detection to the tracker as class 0 (class-agnostic, as `tether-drone-score` counts hits), unfiltered: the tracker applies its own cutoff and band.
- Scores ID kept on every frame where the drone is labeled (`id_kept` with every labeled frame as given). Then it runs the same clips' labeled boxes, with nothing deleted, through the same tracker command, and scores them the same way.
- Tested: a folder that copies the labels scores the same as the labeled-box mode; a detector that finds nothing scores 0 while the labeled-box row is unchanged; missed frames lower the score; a drone seen as two alternating classes keeps one ID (`python/tests/test_antiuav_detections.py`). The band rule's tests are `python/tests/test_low_score_band.py` and `cpp/tests/test_low_score_band.cpp`. Mutation check: `docs/spec-verify/issue-44.md`.

## Reproduce

```
tether-antiuav data/Anti-UAV300 --detections results/antiuav-trt/detections/yolo26n_drone_1280x736_e19/val \
    --work-dir results/antiuav-e2e/drone-at-0.352 --name drone-at-0.352 \
    --tracker "python -m tether.cli --min-score 0.3524462878704071"
tether-antiuav data/Anti-UAV300 --detections results/antiuav-trt/detections/yolo26n_1280x736/val \
    --work-dir results/antiuav-e2e/coco-at-0.25 --name coco-at-0.25 --tracker "python -m tether.cli --min-score 0.25"
```

The C++ row (#46), with the rebuilt engine's dump:

```
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_drone_1280x736.onnx --split val --det-min-score 0.01     --work-dir results/antiuav-trt --detector '"cpp/build/Release/tether-pipeline.exe" --runtime tensorrt --prep cuda'
tether-antiuav data/Anti-UAV300 --detections results/antiuav-trt/detections/yolo26n_drone_1280x736/val     --work-dir results/antiuav-e2e/drone-cpp-at-0.352 --name drone-cpp-at-0.352     --tracker '"cpp/build/Release/tether-track.exe" --min-score 0.3524462878704071'
```

The other rows change `--min-score` (0.25; 0.06577478349208832) and add `--low-score-ratio` (1 for "off"; 0.28373430426985226 for the fixed floor). Each row took about 5.5 minutes, six running at once. The detection dumps are #43's (see [drone-model](drone-model.md) and [drone-coco-baseline](drone-coco-baseline.md) for the commands). The reachable ceiling is `id_kept_ceiling(clip.boxes, 30)` from `python/tether/antiuav.py`, summed over the split's val clips.
