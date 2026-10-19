# MOT17: Tether vs. ByteTrack

Run 2026-10-18 for #7; Tether rows rerun 2026-10-18 for #23, after it began matching confirmed tracks before tentative ones (Tether before #23: 35.6 HOTA / 40.5 IDF1 / 1270 ID switches at 1280×736, 27.8 / 29.2 / 948 at 640), again for #26, after it gained ByteTrack's low-score pass (Tether before #26: 40.9 / 48.6 / 779 at 1280×736, 31.7 / 34.9 / 626 at 640), and again for #35, after the capped search gate went on by default (Tether before #35: 41.3 / 49.0 / 748 at 1280×736, 32.8 / 36.9 / 557 at 640). Both trackers read the same unfiltered detection files (every detection down to score 0.1) and apply their own thresholds. All 7 MOT17 train videos (FRCNN copies), scored by `tether-mot17` with TrackEval 1.3.0, MOT17's official preprocessing, no output filtering for either tracker.

## Results

| Tracker | Detections | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|---|
| Tether | YOLO26n @ 1280×736 | 41.3 | 49.2 | 733 | 55.1% | 85.4% |
| ByteTrack | YOLO26n @ 1280×736 | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| Tether | YOLO26n @ 640 | 32.9 | 37.3 | 531 | 35.0% | 87.2% |
| ByteTrack | YOLO26n @ 640 | 33.5 | 37.9 | 549 | 33.9% | 87.8% |

The 1280×736 rows are the headline (ADR 0002). Recall and precision are CLEAR's, at IoU 0.5.

**Against the SPEC targets** (HOTA within 2 points, IDF1 and ID switches at least as good):

- **1280×736:** all three are met: HOTA 41.33 vs 40.86, IDF1 49.23 vs 48.63, ID switches 733 vs 800.
- **640:** HOTA (0.5 behind) and ID switches (531 vs 549) are met, but IDF1 is 0.6 lower (37.30 vs 37.91). Before #35, Tether also made 8 more ID switches than ByteTrack here (557).

HOTA per video:

| Tracker | Detections | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|
| Tether | YOLO26n @ 1280×736 | 27.5 | 45.6 | 42.0 | 42.9 | 35.8 | 48.9 | 37.7 |
| ByteTrack | YOLO26n @ 1280×736 | 31.2 | 43.9 | 41.9 | 46.3 | 33.2 | 51.1 | 35.1 |
| Tether | YOLO26n @ 640 | 23.9 | 31.5 | 41.9 | 36.9 | 30.7 | 48.1 | 27.8 |
| ByteTrack | YOLO26n @ 640 | 28.0 | 31.8 | 43.2 | 42.9 | 26.1 | 47.6 | 26.7 |

## Where Tether lost: matching order, not low-score detections

Before #23, Tether was about 5 HOTA, 8 IDF1 and 470 ID switches behind ByteTrack at 1280×736. Two diagnostics located the gap.

First, a run of ByteTrack with its second pass turned off (`tether-bytetrack --no-low-score-pass`, which raises `track_low_thresh` to 0.25, so it sees exactly the detections Tether uses):

| Tracker | Detections | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|---|
| ByteTrack | YOLO26n @ 1280×736 | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| ByteTrack, no low-score pass | YOLO26n @ 1280×736 | 40.7 | 48.4 | 889 | 51.6% | 88.5% |
| Tether before #26 (no low-score pass) | YOLO26n @ 1280×736 | 40.9 | 48.6 | 779 | 52.6% | 88.2% |
| ByteTrack | YOLO26n @ 640 | 33.5 | 37.9 | 549 | 33.9% | 87.8% |
| ByteTrack, no low-score pass | YOLO26n @ 640 | 32.8 | 36.6 | 577 | 31.6% | 89.7% |
| Tether before #26 (no low-score pass) | YOLO26n @ 640 | 31.7 | 34.9 | 626 | 32.2% | 89.2% |

- **The low-score pass is worth 0.2 HOTA at 1280×736 and 0.7 at 640.** That was well under a fifth of the old gap, so the gap was not about low-score detections.
- **Tether found as many boxes as ByteTrack does on the same input, but kept fewer IDs.** Before #23 it made 1270 ID switches to the no-low-score ByteTrack's 889, at similar recall and precision.

Second, the association ablations (#22, [mot17-association-ablations.md](mot17-association-ablations.md)) found the cause. Tether matched tentative tracks in the same pass as confirmed ones, so a new tentative track (say, from a duplicate box) could take the detection of the confirmed track it overlapped, and was then confirmed under a new ID. That happened 847 times at 1280×736. Changing the gate, the Kalman model, the number of confirmation frames, the reported boxes or the assignment objective did not help. Matching confirmed tracks first, as ByteTrack does, closed the gap at 1280×736, and that is what Tether does now.

Then #26 added ByteTrack's low-score pass ([mot17-lowscore-ocsort-ablations.md](mot17-lowscore-ocsort-ablations.md) tested it first): tracks that were tracked last frame and found no detection scored ≥ 0.25 can match one scored between 0.1 and 0.25 at IoU ≥ 0.5, and those detections never start a track. The values are ByteTrack's, not tuned. It gave Tether +0.4 HOTA, +0.4 IDF1 and 31 fewer ID switches at 1280×736, and +1.1 HOTA, +2.0 IDF1 and 69 fewer ID switches at 640, the same numbers the #25 prototype measured.

At 640 Tether is still 0.6 HOTA, 1.0 IDF1 and 8 ID switches behind ByteTrack; the cause is not known yet. The ablations found that ByteTrack's Kalman filter on top of the new matching order gives +0.8 HOTA and +2.2 IDF1 at 1280×736, but 172 more ID switches, so it is a separate decision.

The detector has more to give in the low band: at score ≥ 0.1, the 1280×736 set reaches 61.7% recall (67.6% precision), against 54.6% at ≥ 0.25. ByteTrack's second pass only converts about 2.4 points of that, and Tether's the same (52.6% to 55.0% tracked recall).

## ByteTrack settings

Ultralytics' BYTETracker (`ultralytics==8.4.181`, `lap==0.5.12`), run by `tether-bytetrack` (`python/tether/bytetrack.py`), with Ultralytics' defaults for YOLO detectors:

| Setting | Value | Meaning |
|---|---|---|
| `track_high_thresh` | 0.25 | first matching pass uses detections scored at least this (Tether's `min_score` is also 0.25) |
| `track_low_thresh` | 0.1 | second pass uses detections above this and below the high threshold (it matches on plain IoU, at least 0.5) |
| `new_track_thresh` | 0.25 | an unmatched detection starts a track only if scored at least this |
| `track_buffer` | 30 | frames a lost track is kept (Tether's `max_missing_frames` is also 30) |
| `match_thresh` | 0.8 | largest first-pass cost, 1 − IoU × score |
| `fuse_score` | true | first-pass cost weighs IoU by detection score |

- **Not the paper's MOT17 thresholds.** The ByteTrack paper matches high at 0.6 and starts tracks at 0.7, tuned for a YOLOX detector trained on MOT17 and CrowdHuman. YOLO26's scores run much lower; at 0.6 most true people could never start a track.
- **People only.** ByteTrack tracks one class, so it's given the person detections. That's a class filter, not a confidence filter: it still gets every person detection down to 0.1. Tether gets the whole file and only matches same-class pairs.
- **No output filtering.** ByteTrack's own MOT17 script drops boxes wider than 1.6× their height and under 10 px²; that's post-processing, not tracking, so neither tracker gets it.
- Track boxes are ByteTrack's Kalman-filtered boxes; Tether reports the matched detection's box.
- Both match confirmed tracks before new (unconfirmed) ones.
- Both give still-unmatched tracks a second pass against detections scored between 0.1 and 0.25, at IoU ≥ 0.5 (Tether's `LOW_SCORE` and `LOW_MIN_IOU`). Only tracks matched last frame get it, and those detections never start a track.

## Detector ceiling (YOLO26n @ 1280×736)

Detections alone, person class, score ≥ 0.25, scored the same way (`tools/detector_recall.py`). No tracker that reports only detected boxes can find more people.

| Video | 1280×736 recall | 1280×736 precision | 640 recall | 640 precision |
|---|---|---|---|---|
| MOT17-02 | 36.1% | 81.1% | 23.0% | 85.2% |
| MOT17-04 | 57.9% | 87.3% | 24.7% | 90.6% |
| MOT17-05 | 63.7% | 79.9% | 61.5% | 78.9% |
| MOT17-09 | 71.8% | 84.8% | 66.1% | 84.1% |
| MOT17-10 | 57.4% | 84.8% | 39.3% | 81.4% |
| MOT17-11 | 64.8% | 77.2% | 60.1% | 85.1% |
| MOT17-13 | 46.1% | 84.8% | 28.5% | 85.8% |
| All 7 | 54.6% | 84.3% | 33.7% | 85.4% |

## Reproduce

```
pip install -e .[demo,eval,bytetrack]
python tools/export_model.py --model n --size 1280x736
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name tether_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name bytetrack_yolo26n_1280x736 --tracker tether-bytetrack
tether-mot17 data/MOT17 --name tether_yolo26n
tether-mot17 data/MOT17 --name bytetrack_yolo26n --tracker tether-bytetrack
python tools/detector_recall.py data/MOT17 results/mot17/detections/yolo26n_1280x736
python tools/detector_recall.py data/MOT17 results/mot17/detections/yolo26n
```

Diagnostics (the no-low-score-pass rows, and recall at score ≥ 0.1):

```
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name bytetrack-nolow_yolo26n_1280x736 --tracker "tether-bytetrack --no-low-score-pass"
tether-mot17 data/MOT17 --name bytetrack-nolow_yolo26n --tracker "tether-bytetrack --no-low-score-pass"
python tools/detector_recall.py data/MOT17 results/mot17/detections/yolo26n_1280x736 --min-score 0.1 --work-dir results/mot17/min_score_0.1
```

Each run saves `results/mot17/trackers/<name>/results.json`, which names its detection set.
