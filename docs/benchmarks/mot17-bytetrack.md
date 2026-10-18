# MOT17: Tether vs. ByteTrack

Run 2026-10-18 for #7. Both trackers read the same unfiltered detection files (every detection down to score 0.1) and apply their own thresholds. All 7 MOT17 train videos (FRCNN copies), scored by `tether-mot17` with TrackEval 1.3.0, MOT17's official preprocessing, no output filtering for either tracker.

## Results

| Tracker | Detections | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|---|
| Tether | YOLO26n @ 1280×736 | 35.6 | 40.5 | 1270 | 52.5% | 87.8% |
| ByteTrack | YOLO26n @ 1280×736 | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| Tether | YOLO26n @ 640 | 27.8 | 29.2 | 948 | 32.1% | 88.7% |
| ByteTrack | YOLO26n @ 640 | 33.5 | 37.9 | 549 | 33.9% | 87.8% |

The 1280×736 rows are the headline (ADR 0002). Recall and precision are CLEAR's, at IoU 0.5.

**Against the SPEC targets** (HOTA within 2 points, IDF1 and ID switches at least as good), Tether misses all three on both detection sets: about 5 HOTA, 8 IDF1 and 400–470 ID switches behind.

HOTA per video:

| Tracker | Detections | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|
| Tether | YOLO26n @ 1280×736 | 23.9 | 36.2 | 39.6 | 39.4 | 32.6 | 47.0 | 34.9 |
| ByteTrack | YOLO26n @ 1280×736 | 31.2 | 43.9 | 41.9 | 46.3 | 33.2 | 51.1 | 35.1 |
| Tether | YOLO26n @ 640 | 17.9 | 25.9 | 37.0 | 36.2 | 23.6 | 41.9 | 25.2 |
| ByteTrack | YOLO26n @ 640 | 28.0 | 31.8 | 43.2 | 42.9 | 26.1 | 47.6 | 26.7 |

## Where Tether loses: association, not low-score detections

A diagnostic run of ByteTrack with its second pass turned off (`tether-bytetrack --no-low-score-pass`, which raises `track_low_thresh` to 0.25, so it sees exactly the detections Tether uses):

| Tracker | Detections | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|---|
| ByteTrack | YOLO26n @ 1280×736 | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| ByteTrack, no low-score pass | YOLO26n @ 1280×736 | 40.7 | 48.4 | 889 | 51.6% | 88.5% |
| Tether | YOLO26n @ 1280×736 | 35.6 | 40.5 | 1270 | 52.5% | 87.8% |
| ByteTrack | YOLO26n @ 640 | 33.5 | 37.9 | 549 | 33.9% | 87.8% |
| ByteTrack, no low-score pass | YOLO26n @ 640 | 32.8 | 36.6 | 577 | 31.6% | 89.7% |
| Tether | YOLO26n @ 640 | 27.8 | 29.2 | 948 | 32.1% | 88.7% |

- **The low-score pass is worth 0.2 HOTA at 1280×736 and 0.7 at 640.** It accounts for well under a fifth of the gap, so a low-score pass for Tether is not where to start.
- **Tether finds as many boxes as ByteTrack does on the same input** (recall 52.5% vs 51.6%, similar precision), but makes 1270 ID switches to 889, and IDF1 is 8 points lower. The gap is in keeping IDs, not in finding people.
- Why is not known yet. Candidates, all untested: the match gate (Tether: IoU ≥ 0.3 against the predicted box; ByteTrack: score-weighted IoU, which accepts IoU down to 0.2 for a confident box), lost tracks (ByteTrack keeps matching them by IoU to the predicted box for 30 frames; Tether's searching tracks do the same, but the motion model differs), and confirmation (3 frames for Tether, 2 for ByteTrack). Finding out is its own issue.

The detector has more to give in the low band: at score ≥ 0.1, the 1280×736 set reaches 61.7% recall (67.6% precision), against 54.6% at ≥ 0.25. ByteTrack's second pass only converts about 2.4 points of that.

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
