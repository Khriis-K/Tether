# MOT17: why Tether loses IDs that ByteTrack keeps

Run 2026-10-18 for #22. Same setup as [mot17-bytetrack.md](mot17-bytetrack.md): YOLO26n detections already dumped, all 7 MOT17 train videos, TrackEval via `tether-mot17`. Each row swaps one ByteTrack behaviour into Tether's tracker and leaves everything else at Tether's defaults. Tether's defaults were not changed; the ablations ran through a throwaway copy of `Tracker.update`, which with no flags gives byte-identical track files to `tether-track`.

## Answer: matching order

Tether matches every track in one Hungarian pass, tentative tracks included. ByteTrack matches confirmed tracks (tracked and lost) first. Unconfirmed tracks only get the detections left over after that.

In one pass, a new tentative track can take a detection from the confirmed track it sits on, for example one started by a duplicate box. The confirmed track drops to searching, the tentative one is confirmed under a new ID, and that is an ID switch. On the 1280×736 detections this happens 847 times across the 7 videos: a tentative track wins a detection that a confirmed track, left unmatched that frame, also overlapped at IoU ≥ 0.3.

Matching confirmed tracks first, and nothing else changed, closes the gap at 1280×736:

| Tracker | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|
| Tether | 35.6 | 40.5 | 1270 | 52.5% | 87.8% |
| Tether, confirmed tracks matched first | 40.9 | 48.6 | 779 | 52.6% | 88.2% |
| ByteTrack | 40.9 | 48.6 | 800 | 54.0% | 86.1% |
| ByteTrack, no low-score pass | 40.7 | 48.4 | 889 | 51.6% | 88.5% |

Against the SPEC targets at 1280×736: HOTA is level (40.92 vs 40.86) and ID switches beat ByteTrack (779 vs 800). IDF1 is 0.03 short (48.60 vs 48.63).

HOTA per video:

| Tracker | 02 | 04 | 05 | 09 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|
| Tether | 23.9 | 36.2 | 39.6 | 39.4 | 32.6 | 47.0 | 34.9 |
| Tether, confirmed first | 26.9 | 45.2 | 42.8 | 43.9 | 33.7 | 49.6 | 36.2 |
| ByteTrack | 31.2 | 43.9 | 41.9 | 46.3 | 33.2 | 51.1 | 35.1 |

At 640 the ordering also helps, but it leaves a gap:

| Tracker (YOLO26n @ 640) | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|
| Tether | 27.8 | 29.2 | 948 | 32.1% | 88.7% |
| Tether, confirmed tracks matched first | 31.7 | 34.9 | 626 | 32.2% | 89.2% |
| ByteTrack | 33.5 | 37.9 | 549 | 33.9% | 87.8% |
| ByteTrack, no low-score pass | 32.8 | 36.6 | 577 | 31.6% | 89.7% |

That puts HOTA within the 2-point target (1.8 behind), but IDF1 is still 3.0 behind and ID switches 77 worse.

## The other candidates, one at a time (1280×736)

| Change from Tether's defaults | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|
| none (Tether) | 35.6 | 40.5 | 1270 | 52.5% | 87.8% |
| confirmed tracks matched first | **40.9** | **48.6** | **779** | 52.6% | 88.2% |
| match gate: cost 1 − IoU × score ≤ 0.8 | 34.8 | 38.8 | 1306 | 51.4% | 88.7% |
| assignment: lowest total cost, as `lap.lapjv` (Tether maximises the number of pairs first) | 35.7 | 40.7 | 1263 | 52.5% | 87.8% |
| motion: ByteTrack's XYAH Kalman filter | 35.9 | 42.2 | 1576 | 52.7% | 87.4% |
| motion: Tether's filter, center-velocity noise 1/160 (ByteTrack's) instead of 1/10 | 35.3 | 40.7 | 1413 | 52.6% | 87.6% |
| confirmation after 2 frames instead of 3 | 35.6 | 40.1 | 1675 | 53.7% | 86.5% |
| report the Kalman-filtered box instead of the detection | 35.6 | 40.5 | 1271 | 52.6% | 87.9% |
| no center-distance second chance for once-seen tracks | 35.4 | 40.1 | 1302 | 52.5% | 87.7% |
| all of the above together | 40.3 | 47.9 | 950 | 51.6% | 88.3% |

- **The fused gate is not "looser".** It accepts IoU down to 0.2 only for a box scored 1.0. At Tether's 0.25 cutoff it needs IoU ≥ 0.8. With YOLO26's low scores it is mostly stricter, so it loses recall.
- **Confirming after 2 frames hurts.** That fits the mechanism: tentative tracks that steal detections get confirmed sooner.
- **Reporting the filtered box changes nothing measurable.**

## On top of the new ordering (1280×736)

| Confirmed first, plus | HOTA | IDF1 | ID switches | Recall | Precision |
|---|---|---|---|---|---|
| nothing | 40.9 | 48.6 | 779 | 52.6% | 88.2% |
| fused gate (ByteTrack's 0.8 / 0.7 limits) | 40.7 | 47.9 | 681 | 51.0% | 89.5% |
| lowest-total-cost assignment | 41.0 | 48.7 | 764 | 52.6% | 88.2% |
| ByteTrack's XYAH Kalman filter | 41.7 | 50.8 | 951 | 52.9% | 87.9% |
| confirmation after 2 frames | 39.7 | 46.3 | 1216 | 53.7% | 86.6% |
| Kalman-filtered box reported | 41.0 | 48.7 | 783 | 52.7% | 88.3% |
| no center-distance second chance | 40.8 | 48.5 | 799 | 52.7% | 88.1% |

ByteTrack's Kalman filter is the only addition with a sizeable effect, and it pulls two ways: +0.8 HOTA and +2.2 IDF1, but 172 more ID switches. Why is not known yet. (IDF1 rewards following each person with one ID for longer, while IDSW counts every change of ID, so the filter may re-find more lost tracks and also mis-assign more of them.)

## Recommendation

A fix issue: **match confirmed (tracked and searching) tracks first, then tentative tracks against the leftover detections.** Keep the IoU ≥ 0.3 gate, the 3-frame confirmation and the reported boxes as they are. This is one change with the largest measured effect, and the track lifecycle (tentative → tracked → searching) stays as it is. After it, compare the motion model as its own follow-up, since that trade-off needs a decision. Also worth one line in the fix issue: at 640, IDF1 and ID switches still miss the SPEC target after the reordering.

## Reproduce

The ablation tracker was a throwaway script and is not in the repo. The confirmed-first row is reproduced by the fix itself. Reference rows:

```
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name tether_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name bytetrack_yolo26n_1280x736 --tracker tether-bytetrack
```
