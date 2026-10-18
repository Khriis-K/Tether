# MOT17: low-score pass and OC-SORT on top of confirmed-first matching

Run 2026-10-18 for #25. Same setup as [mot17-bytetrack.md](mot17-bytetrack.md): YOLO26n detections already dumped, all 7 MOT17 train videos, TrackEval via `tether-mot17`. The baseline is main at 88d313e (confirmed tracks matched first, #23). Each row adds one piece through a throwaway copy of `Tracker.update`; with no flags, that copy gives byte-identical track files to `tether-track`. Tether's defaults were not changed.

Every setting is a published default, not tuned here, because all of these numbers come from the same 7 videos:

- **Low-score pass (ByteTrack):** detections scored above 0.1 and below Tether's 0.25 cutoff are matched, at IoU ≥ 0.5, only to confirmed tracks that were tracked last frame and are still unmatched after the first pass. They never start tracks.
- **OCR (OC-SORT's observation-centric recovery):** confirmed tracks still unmatched get one more IoU ≥ 0.3 pass against the leftover detections, using their last observed box instead of the prediction.
- **OCM (OC-SORT's observation-centric momentum):** the IoU cost gets a direction-consistency term with weight 0.2, scaled by detection score. The track's direction comes from observations Δt = 3 frames apart.
- **ORU (OC-SORT's observation-centric re-update):** when a searching track is re-found, its Kalman filter is restored to its last observation and replayed along a straight line to the new one.

## Results

| Tracker | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|
| **YOLO26n @ 1280×736** | | | | | |
| Tether (main) | 40.9 | 48.6 | 779 | 52.6% | 88.2% |
| + low-score pass | **41.3** | **49.0** | **748** | 55.0% | 85.7% |
| + OCR | 41.5 | 50.0 | 735 | 53.0% | 87.7% |
| + OCM | 40.8 | 48.3 | 792 | 52.6% | 88.2% |
| + ORU | 41.2 | 48.9 | 781 | 52.7% | 88.2% |
| + low-score pass + OCR | 41.4 | 49.3 | 737 | 55.5% | 85.1% |
| + low-score pass + OCR + ORU | 41.4 | 49.4 | 748 | 55.5% | 85.0% |
| + all four | 41.3 | 49.7 | 743 | 55.5% | 85.1% |
| ByteTrack | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| **YOLO26n @ 640** | | | | | |
| Tether (main) | 31.7 | 34.9 | 626 | 32.2% | 89.2% |
| + low-score pass | **32.8** | **36.9** | **557** | 34.8% | 87.3% |
| + OCR | 32.0 | 35.5 | 636 | 32.4% | 88.7% |
| + OCM | 31.2 | 34.1 | 615 | 32.2% | 89.2% |
| + ORU | 31.7 | 34.9 | 613 | 32.2% | 89.2% |
| + low-score pass + OCR | 32.4 | 37.0 | 560 | 35.1% | 86.7% |
| + low-score pass + OCR + ORU | 32.7 | 37.3 | 564 | 35.1% | 86.7% |
| + all four | 32.5 | 36.9 | 555 | 35.1% | 86.7% |
| ByteTrack | 33.5 | 37.9 | 549 | 33.9% | 87.8% |

HOTA per video:

| Tracker | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| **1280×736** | | | | | | | |
| Tether (main) | 26.9 | 45.2 | 42.8 | 43.9 | 33.7 | 49.6 | 36.2 |
| + low-score pass | 27.3 | 45.3 | 42.8 | 44.7 | 35.6 | 49.0 | 37.5 |
| + OCR | 29.4 | 45.2 | 42.6 | 41.8 | 34.9 | 52.3 | 36.2 |
| ByteTrack | 31.2 | 43.9 | 41.9 | 46.3 | 33.2 | 51.1 | 35.1 |
| **640** | | | | | | | |
| Tether (main) | 21.8 | 30.4 | 41.7 | 38.0 | 27.3 | 47.6 | 26.6 |
| + low-score pass | 23.1 | 31.4 | 41.8 | 39.0 | 30.0 | 47.9 | 27.8 |
| + OCR | 22.2 | 30.6 | 40.8 | 36.6 | 28.1 | 49.0 | 26.5 |
| ByteTrack | 28.0 | 31.8 | 43.2 | 42.9 | 26.1 | 47.6 | 26.7 |

## What it shows

- **The low-score pass is the one consistent gain.** It improves HOTA, IDF1 and ID switches at both input sizes. HOTA rises on 6 of 7 videos at 1280×736 (MOT17-11 drops 0.6) and on all 7 at 640.
- **With it, Tether meets all three SPEC targets at 1280×736:** HOTA 41.29 vs 40.86, IDF1 49.04 vs 48.63, and 748 ID switches vs 800.
- **At 640 it still misses two of the three targets.** HOTA is within the 2-point target (32.8 vs 33.5), but IDF1 is 1.0 behind and ID switches are 8 worse (557 vs 549). The remaining gap is mostly on MOT17-02 and MOT17-09. Why is not known.
- **OCR is not a robust gain.** At 1280×736 it has the best single-row HOTA and IDF1, but per video it is +2.5 on 02 and +2.7 on 11, and −2.1 on 09. At 640 it adds 0.2 HOTA and 10 ID switches. On top of the low-score pass it adds nothing measurable at 1280×736 (HOTA 41.29 → 41.36) and costs 0.4 HOTA at 640 (32.82 → 32.42). Gains that swing by video like this are not a mechanism to rely on.
- **OCM is slightly negative at both sizes.**
- **ORU is near zero.** It changes HOTA by at most +0.3, and ID switches by fewer than 15 either way.

## Recommendation

A fix issue for the **low-score pass only**, with ByteTrack's published settings. Leave OCR, OCM and ORU out of the tracker that gets frozen for the C++ port. If the 640 gap matters later, look at MOT17-02 and MOT17-09 first.

## Reproduce

The ablation tracker was a throwaway script and is not in the repo. The low-score-pass rows are reproduced by the fix. The baseline and reference rows:

```
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name tether_yolo26n_1280x736
tether-mot17 data/MOT17 --name tether_yolo26n
```
