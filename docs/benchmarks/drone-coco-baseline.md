# Drone detection: the COCO "before" row on the leak-free val

Run 2026-10-21 for #42. This is the pretrained COCO **YOLO26n at 1280×736** (`models/yolo26n_1280x736.onnx`, the shipped detector, ADR 0003), scored **class-agnostically**: any detection counts as "drone". It's run on the **new session-disjoint Anti-UAV300 val** (`splits/antiuav.json`, seed 0) and is the row every drone-model number is compared with ([ADR 0004](../adr/0004-drone-detector-on-anti-uav.md)). Nothing was trained on these frames. Anti-UAV test and Seraphim weren't touched.

*Update (2026-10-23, #48): on Anti-UAV test, at the same TensorRT runtime and 0.01 floor, the COCO model's AP50 is 0.402; at 0.25 recall is 0.222 and precision 0.757 (`drone-test-coco.json`, [drone-test](drone-test.md)).*

Raw results: [`drone-coco-baseline.json`](drone-coco-baseline.json) (from `tether-drone-score`), and [`drone-coco-baseline-breakdown.json`](drone-coco-baseline-breakdown.json) for the by-class and by-session tables (from `tools/drone_breakdown.py`).

## The val

Official train + val (227 clips, 211,527 frames, 27 recording sessions) were pooled. Five whole sessions were held out with seed 0, alternating between the two recording days until they held at least 15% of the frames (`tether-antiuav-split`):

| | Clips | Frames | Sessions |
|---|---|---|---|
| train | 185 | 172,045 | 22 |
| **val** | **42** | **39,482 (18.7% of pooled)** | **5**: 20190925_131530, 20190925_133630, 20190925_205804, 20190926_142435, 20190926_143632 |
| test (official, untouched) | 91 | 85,374 | 10 |

In val the drone is labeled on 38,250 frames and absent on 1,232. All clips are 1920×1080 at 20 fps.

## Results

Model: `yolo26n_1280x736`, the pretrained COCO weights (no drone class). AP50 is **0.146** (scores ≥ 0.1, the dump's floor).

| Threshold | Chosen by | Recall | Precision | F1 | False alarms / min (drone absent) |
|---|---|---|---|---|---|
| 0.100 | best F1 on val, at the floor (`at_lowest_saved_score` in the JSON; see caveats) | 0.312 (11,944 of 38,250) | 0.309 | 0.311 | 76.0 (78 in 1.03 min) |
| 0.25 | the tracker's min score | 0.142 (5,414 of 38,250) | 0.410 | 0.210 | 7.8 (8 in 1.03 min) |

Recall by labeled box size, at source resolution (area: tiny < 16², small < 32², medium < 96², large ≥ 96² px):

| Size | Drones | Recall at 0.100 | Recall at 0.25 |
|---|---|---|---|
| tiny | 8 | 0.000 | 0.000 |
| small | 12 | 0.417 | 0.333 |
| medium | 20,530 | 0.274 | 0.127 |
| large | 17,700 | 0.356 | 0.158 |

"small" here is 16²–32², not COCO's "everything under 32²", so don't compare it with COCO's AP_small.

Which COCO class found the drone (the best-scored hit on each frame where it was found):

| | airplane | traffic light | person | boat | car | bird | others | total found |
|---|---|---|---|---|---|---|---|---|
| at 0.100 | 6,736 | 2,059 | 1,691 | 758 | 504 | 160 | 36 (cow 16, truck 15, bench 2, skateboard 2, bear 1) | 11,944 |
| at 0.25 | 4,083 | 439 | 652 | 84 | 112 | 42 | 2 (cow) | 5,414 |

Recall by val session, over the frames where the drone is labeled:

| Session | Drone frames | Recall at 0.100 | Recall at 0.25 |
|---|---|---|---|
| 20190925_131530 | 6,889 | 0.066 | 0.011 |
| 20190925_133630 | 9,443 | 0.196 | 0.071 |
| 20190925_205804 | 6,927 | 0.184 | 0.039 |
| 20190926_142435 | 8,036 | 0.520 | 0.219 |
| 20190926_143632 | 6,955 | 0.601 | 0.378 |

## TensorRT at a 0.01 floor (#43)

Redumped 2026-10-21 for #43 so the curve isn't cut off at 0.1: the same model and val, on the C++ pipeline's TensorRT runtime (`cpp/build-trt`, TensorRT 11.4.0.106, CUDA 13.4, engine `models/yolo26n_1280x736.engine` built with `trtexec --noTF32`, `--runtime tensorrt --prep cuda`, sequential), saving scores ≥ **0.01**. Every drone-model val number is dumped the same way, at the same floor, so this is the row they're compared with. Raw results: [`drone-coco-baseline-trt.json`](drone-coco-baseline-trt.json), and [`drone-coco-baseline-trt-at-0.1.json`](drone-coco-baseline-trt-at-0.1.json) for the 0.100 row (`--threshold 0.1`).

AP50 is **0.228** (scores ≥ 0.01). The best-F1 cut is **0.066, no longer at the floor** (`at_lowest_saved_score: false`): F1 peaks above 0.01 and falls below it.

| Runtime, floor | Threshold | Chosen by | Recall | Precision | F1 | False alarms / min (drone absent) |
|---|---|---|---|---|---|---|
| TensorRT, 0.01 | 0.066 | best F1 on val | 0.397 (15,194 of 38,250) | 0.270 | 0.321 | 137.3 (141 in 1.03 min) |
| TensorRT, 0.01 | 0.100 | given, to compare with the CPU row | 0.312 (11,944 of 38,250) | 0.309 | 0.311 | 76.0 (78 in 1.03 min) |
| TensorRT, 0.01 | 0.25 | the tracker's min score | 0.142 (5,414 of 38,250) | 0.410 | 0.210 | 7.8 (8 in 1.03 min) |
| ONNX Runtime CPU, 0.1 (above) | 0.100 | best F1 on val, at the floor | 0.312 (11,944 of 38,250) | 0.309 | 0.311 | 76.0 (78 in 1.03 min) |
| ONNX Runtime CPU, 0.1 (above) | 0.25 | the tracker's min score | 0.142 (5,414 of 38,250) | 0.410 | 0.210 | 7.8 (8 in 1.03 min) |

- **The runtime doesn't move the rows.** At 0.1 and at 0.25 TensorRT finds the same drones and the same false alarms as the CPU run. At 0.1 it keeps 38,676 detections against the CPU's 38,674, so two scores sit on different sides of 0.1; neither is a hit or a false alarm on an absent frame.
- **The lower floor moves AP50 from 0.146 to 0.228**, all from detections scored 0.01–0.1. The two AP50s aren't comparable with each other, only with dumps at their own floor. The 0.228 is still a lower bound over a full curve, but the best-F1 cut is now inside the saved range.
- **The best-F1 cut buys recall with false alarms.** At 0.066 the COCO model finds 39.7% of drones at 27.0% precision, with 137 false alarms per minute on the drone-absent frames.
- Recall by box size at 0.066: medium 0.351 (7,201 of 20,530), large 0.451 (7,987 of 17,700), small 6 of 12, tiny 0 of 8.
- The dump took about 5.5 minutes (42 clips), against about an hour on the CPU.

## What it means

- **The COCO model mostly misses the drone.** At the tracker's 0.25 it finds 14.2% of labeled drones. Even at the 0.1 floor it finds under a third. Most hits come in as "airplane", and "bird" barely shows up.
- **Box size isn't the reason.** Medium and large boxes, nearly all of val, are found at 0.127 and 0.158 at 0.25. The ADR's evidence that the drones aren't tiny at 1280×736 holds. This doesn't say whether the misses are no box at all or a box that overlaps under IoU 0.5; that wasn't measured.
- **Per-session spread is large** (1.1% to 37.8% recall at 0.25). With only five sessions in val, a later model's val score will carry that spread too. Compare models on the same split, and don't read small AP50 differences as real.

## Caveats

- **The best-F1 cut sits on the floor** (in the CPU run; the TensorRT run at a 0.01 floor above isn't). Detections were saved at score ≥ 0.1, and F1 was still rising at 0.1, so the true best-F1 threshold for this model is likely lower. The same floor truncates the precision-recall curve, which makes **AP50 0.146 a lower bound** over the full curve. It's comparable with other dumps made at the same 0.1 floor. The floor matches the tracker's low-score band (0.1–0.25), and it's recorded in each dump's `dump.json` and the results file.
- **The false-alarm rate rests on about one minute.** Val has 1,232 drone-absent frames (1.03 min). The 0.25 row is 8 detections. Treat it as an order of magnitude, not a rate to compare to one decimal. The ADR's "empty sky" row will be noisy for the same reason on any split of this dataset. It counts detections, not frames, so a frame with two boxes counts twice. And these are easy frames: a low rate on empty sky doesn't show the model can tell a distant bird from a distant drone.
- **Class-agnostic is generous to COCO in one way and harsh in another.** Any class counts, so an "airplane" or "traffic light" box on the drone is a hit. But YOLO26 has no class-agnostic NMS, so two boxes of different classes on the drone count as one hit plus one false positive.
- **Runtime.** Detections came from the C++ pipeline on ONNX Runtime's CPU provider (`tether-pipeline`, the main checkout's build without TensorRT). Its detection files were byte-identical to the Python pipeline's on the two videos checked ([cpp-pipeline-speed](cpp-pipeline-speed.md)), and TensorRT's scores agreed to 0.00032 on the detections checked at ≥ 0.25 ([tensorrt-runtime](tensorrt-runtime.md)). So the runtime shouldn't move the 0.25 row; agreement between 0.1 and 0.25 wasn't checked here; the TensorRT redump above checks it on this val.

## How it's scored

`tether-drone-score` (`python/tether/drone_score.py`):

- **Hit.** IoU ≥ 0.5 with the labeled box, the same rule as the Anti-UAV tracking score. On each frame the drone is found by its best-scored overlapping detection. Every other detection is a false positive, including a second box on the drone (COCO's greedy rule).
- **AP50.** Precision and recall at every distinct score (tied scores cut together), the precision envelope taken, then the area under it.
- **Thresholds.** The best-F1 cut on the split (or a given one, e.g. frozen from val before test), plus the tracker's 0.25. Thresholds below the dump's floor are refused.
- **False alarms per minute.** Detections at or above the threshold on frames where the drone is labeled absent, divided by those frames' duration at each clip's own frame rate.
- The null cases are tested: no detections, and the same box every frame, score AP50 ≈ 0; copying the labels scores 1 (`python/tests/test_drone_score.py`, and `docs/spec-verify/issue-42.md` for the mutation check).

## Reproduce

```
tether-antiuav-split data/Anti-UAV300                      # rewrites splits/antiuav.json; seed 0 gives the committed file
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_1280x736.onnx --split val \
    --detector cpp/build/Release/tether-pipeline.exe       # or omit --detector for tether-demo (slower)
tether-drone-score data/Anti-UAV300 results/antiuav/detections/yolo26n_1280x736/val --split val \
    --out docs/benchmarks/drone-coco-baseline.json
python tools/drone_breakdown.py data/Anti-UAV300 results/antiuav/detections/yolo26n_1280x736/val     --out docs/benchmarks/drone-coco-baseline-breakdown.json
```

The TensorRT row at the 0.01 floor (#43), with the engine built from the ONNX file by `trtexec --noTF32`:

```
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_1280x736.onnx --split val --det-min-score 0.01     --work-dir results/antiuav-trt --detector "cpp/build-trt/Release/tether-pipeline.exe --runtime tensorrt --prep cuda"
tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_1280x736/val --split val     --out docs/benchmarks/drone-coco-baseline-trt.json
```

The CPU dump took about an hour on the laptop CPU. The scorer takes seconds.
