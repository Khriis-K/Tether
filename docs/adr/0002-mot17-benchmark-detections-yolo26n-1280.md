# MOT17 benchmark detections: YOLO26n at 1280 input

The first MOT17 run (#6) scored HOTA 27.8. The tracker's recall (32.1%) sat right at the detector's own ceiling (about 34%): YOLO26n at 640 input missed two-thirds of the people, almost all of the ones under 100 px tall. MOT17 frames are 1920×1080, so letterboxing to 640 shrinks people 3×. With that few people reaching the trackers, a Tether-vs-ByteTrack gap (#7) would say little, and it would tilt toward ByteTrack, whose second matching pass uses the low-score detections a weak detector produces.

**Decision:** #7 headlines detections from **YOLO26n at 1280 input** (as a 1280×736 rectangle, see the update below), and also reports a second row on the **shipped detector (YOLO26n at 640)**. The benchmark detection set and the shipped detector are separate choices: detections are dumped to a file once, so detector speed doesn't affect the benchmark. The shipped detector stays YOLO26n at 640 until #18 measures speed and accuracy together on TensorRT.

## Evidence

Raw detections, no tracker: person class, score ≥ 0.25, Hungarian matching at IoU 0.5 against the ground truth (class 1, active). The 1280 rows used a 1280×1280 square input. The precision column is understated: it counted detections on static people, distractors and reflections as false alarms, which the official evaluation ignores (see the update). MOT17-02 and MOT17-04 only. CPU timings are ONNX Runtime on the dev laptop, median per frame, for prep + inference + decode.

| Detector | Recall 02 | Recall 04 | Precision 02 / 04 | CPU ms/frame |
|---|---|---|---|---|
| n @ 640 (shipped) | 23.0% | 24.7% | 79.8 / 90.6 | 39 |
| s @ 640 | 28.7% | 30.2% | 77.1 / 92.5 | 106 |
| m @ 640 | 31.0% | 29.8% | 83.3 / 92.8 | 312 |
| **n @ 1280** | **37.3%** | **60.4%** | **73.4 / 83.6** | **147** |
| s @ 1280 | 40.3% | 63.1% | 69.8 / 82.9 | 434 |
| n @ 1088×1920 (native) | 39.5% | 63.1% | 72.0 / 75.7 | 189 |

- **Input size is the lever, not model size.** On MOT17-04, people 50–100 px tall go from 0% found at n@640 to 61% at n@1280. A model 3× bigger at 640 (m) still finds 0%.
- **Past 1280, returns flatten.** s@1280 and native resolution add about 3 points each, and both cost precision. At native resolution, large people get slightly worse (02, >200 px: 78% → 74%), which fits a model trained on 640 images seeing people bigger than it was trained on.
- **What's left is out of reach for input size.** People over 200 px tall top out around 75–80% for every variant (most likely occlusion), and people under 50 px stay under 5%.

## Consequences

- **The detection set must be named wherever MOT17 numbers appear.** The SPEC's "within 2 HOTA points of ByteTrack" means something different at 34% detector recall than at 60%.
- **Input size has to become a setting** (it was hardcoded as `INPUT_SIZE = 640` in `python/tether/detector.py`), and it should allow a rectangle: a 1280 square spends about 44% of its pixels on padding for a 16:9 frame. Done in #21: the detector reads its input size from the model.
- **#18 compares model size × input size, not model size alone.** At 640, n/s/m barely differ. Left as written, #18 would conclude "bigger models don't help" and miss the real trade-off.
- **This only covers 2 of the 7 train videos.** The full dump in #7 confirms or corrects it.
- **No detector trained on MOT17 or CrowdHuman.** That's how ByteTrack's published numbers were made, and it would raise the occlusion ceiling, but we evaluate on the MOT17 train split, so a detector trained on those videos would leak. Revisit only with a held-out split.
- **Low-score matching for Tether is not decided here.** m@640 finds 42% of 50–100 px people on MOT17-04, but only at scores 0.1–0.25, which Tether's `MIN_SCORE = 0.25` throws away. If #7 shows Tether losing there, that becomes its own issue.

## Update (2026-10-18): 1280×736, not the square

Once #21 made rectangular inputs work, we compared YOLO26n at 1280×736 with the 1280×1280 square the evidence above used. Both shrink a 1920×1080 frame by 2/3; only the padding differs. The benchmark detection set is now **YOLO26n at 1280×736** (`tools/export_model.py --model n --size 1280x736`, giving `yolo26n_1280x736.onnx`).

Scored with the official MOT17 rule (detections matched to static-person, person-on-vehicle, distractor or reflection boxes are dropped, not counted as false alarms), score ≥ 0.25, MOT17-02 and 04 only:

| Input | Recall 02 / 04 | Precision 02 / 04 | CPU ms/frame |
|---|---|---|---|
| 1280×1280 | 37.2% / 59.9% | 80.1 / 85.7 | 158.6 |
| **1280×736** | **36.0% / 57.9%** | **81.0 / 87.3** | **91.9** |

- **It's 42% cheaper, and not quite free.** At a fixed 0.25 cutoff, 736 trades 1–2 points of recall for 1–2 points of precision, mostly because its scores run slightly lower. At matched precision the square still leads by about 1 recall point.
- **Why the shape matters is only partly understood.** Where the padding falls on the stride-32 grid explains about half the gap on 02 and none of it on 04. A model trained on square letterboxed images may simply be slightly off on a wide canvas; that's untested.
- **The comparison in #7 stays fair either way:** both trackers read the same detection set. Only the absolute HOTA/MOTA move.
- Timing was one run on one machine (ONNX Runtime CPU, prep + inference + decode, median of 300 frames); the ratio matches the pixel ratio (0.575×) and should carry over better than the milliseconds.

## Update (2026-10-18): all 7 videos confirm it

#7 dumped YOLO26n at 1280×736 for all 7 train videos. Scored the same way (official rule, score ≥ 0.25, `tools/detector_recall.py`):

| Detections | Recall, all 7 | Precision, all 7 |
|---|---|---|
| n @ 640 (shipped) | 33.7% | 85.4% |
| **n @ 1280×736** | **54.6%** | **84.3%** |

- **The decision holds.** Recall goes up by 21 points at the same precision. The 2-video numbers above carry over (02 and 04 match to 0.1 point).
- **The gain is uneven.** It's 13–33 points on 02, 04, 10 and 13, and 2–6 points on 05, 09 and 11. MOT17-05 is a 640×480 video, so 640 input already sees it at full size.
- Per-video numbers are in `docs/benchmarks/mot17-bytetrack.md`, along with the tracker comparison: ByteTrack leads Tether by about 5 HOTA on both detection sets, and almost none of that comes from low-score detections, so the low-score question in Consequences is answered: not worth an issue on its own.

## Update (2026-10-21): the shipped detector is now YOLO26n at 1280×736 too

#18 measured speed and accuracy together on TensorRT, and #17 moved frame prep to the GPU. YOLO26n at 1280×736 now ships, so the benchmark detection set and the shipped detector are the same model. See ADR 0003 (`0003-ship-yolo26n-1280x736.md`), which replaces this ADR's "the shipped detector stays YOLO26n at 640".
