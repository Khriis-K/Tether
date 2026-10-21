# Model size × input size on TensorRT: which detector to ship (#18)

**Question:** the spec's speed targets (120+ FPS, P99 frame time under 16 ms, on video files) assume YOLO26n at 640. ADR 0002 found that on MOT17 a bigger input mattered far more than a bigger model. With TensorRT and the CUDA graph in place, what does each YOLO26 model (n, s, m) at each input size (640×640, 1280×736) cost in speed, what does it buy in MOT17 accuracy, and which one should ship?

**Answer: ship YOLO26n at 1280×736.** It's the most accurate row that meets both speed targets on the video file: HOTA 41.55 / IDF1 49.33, against 32.74 / 37.05 for today's YOLO26n at 640 (+8.81 HOTA, +12.27 IDF1). It runs the Anti-UAV clip at **124.82–126.18 FPS, P99 9.17–10.45 ms** in parallel mode. That clears 120 FPS by only 4.82–6.18 FPS. **Frame prep is its slowest stage** (busy 99.2–99.3%, detect next at 51.4–51.8%), so **#17 (CUDA frame prep) becomes a go**, per #15's rule.

The rest of the table:
- **Bigger models do help, at both sizes**, more than ADR 0002's two-video sample suggested. At 1280×736, s adds 3.30 HOTA and m adds 5.34 over n. But neither fits the targets: s runs 108.89–110.06 FPS and m 46.77–47.20, both limited by detect (9.02–9.16 and 21.16–21.36 ms). CUDA prep can't speed up a detect-bound row.
- **YOLO26s at 640 also meets both targets** (149.54–185.11 FPS), with HOTA 37.81, but that's 3.73 below n at 1280×736.
- **No row meets either target on MOT17-04.** It's a folder of JPEGs, and decoding them caps every row at 64.53–66.07 FPS (capture busy 99.8–99.9%), except m at 1280×736, which drops to 47.43–47.49 FPS because its detect takes 21.03–21.06 ms. That's the same decode limit #15 and #40 found.
- **Nothing meets 120 FPS sequentially.** The best is n at 640 on the Anti-UAV clip, at 111.95–112.68 FPS.

## Setup

- **Models:** `tools/export_model.py --model {n,s,m} --size {640,1280x736}`, Ultralytics 8.4.181, opset 18, float32, NMS-free head (`[1,300,6]`). The exact input shape for each row is read from the model by the tool and listed in the tables as width×height: 640×640 or 1280×736 (a 16:9 frame scaled by 2/3, with 8 px of padding top and bottom; see #21). The files are `yolo26n.onnx` (the shipped 640 model), `yolo26n_1280x736.onnx`, `yolo26{s,m}_640x640.onnx` and `yolo26{s,m}_1280x736.onnx`.
- **Engines:** `trtexec --onnx=M.onnx --saveEngine=M.engine --noTF32 --skipInference` with TensorRT 11.4.0.106 on CUDA 13.4. The four s/m builds took 78–90 s each; the n engines are the #16/#40 ones. Engines are gitignored.
- **Runtime:** `tether-pipeline --runtime tensorrt` with the CUDA graph on (the default, #40) and **CPU frame prep**, the only prep path (#17 isn't built). Release build, MSVC, Windows 11, AMD Ryzen AI 7 350, RTX 5070 Laptop GPU (8 GB, driver 617.42), on mains power. Headless.
- **Do the new engines give ONNX Runtime's detections?** Yes. `tether-runtime-compare` on the first 100 frames of MOT17-04, with #16's tolerance, found every detection at score 0.25 or above had a partner for all four s/m engines (3,932 to 10,470 checked, worst score difference 0.00020).
- **Speed:** `tether-model-compare VIDEO --models <all six> --rounds 2 --pipeline {parallel,sequential}` (new for this issue, `python/tether/model_compare.py`). It runs each model twice, rotating the model order each round, and summarizes each timing log with `tether-timing`'s summary plus each step's [busy share](../CONCEPTS.md#busy-share-finding-the-bottleneck-stage). Every frame is timed, including the first. A row meets a target only if **both** rounds met it.
- **Videos:** the Anti-UAV test clip `20190925_111757_1_1/visible.mp4` (1920×1080 H.264, 1000 frames), the "video file" for the targets, as in #14–#40; and MOT17-04 (`img1/%06d.jpg`, 1920×1080 JPEGs, 1050 frames).
- **Accuracy:** all 7 MOT17 train videos (FRCNN copies), detections dumped **with TensorRT** (`tether-pipeline --runtime tensorrt --dump`, every detection down to score 0.1), then scored by the existing `tether-mot17` (TrackEval 1.3.0, the official preprocessing) with the default tracker as committed at 99bc602. Detector recall and precision (score ≥ 0.25, no tracker) come from `tools/detector_recall.py`.
- **Files:** [`model-size.json`](model-size.json) has every run's timing summary and busy shares, the rows, the MOT17 and detector-recall results, the agreement checks, and the earlier n-only runs. Raw timing logs, dumps and tracks are in `results/model-size/` (ignored).

## The combined table (parallel, the fast mode)

Ranges are min–max over two rounds. MOT17 is one number per row: the scoring is deterministic for a given detection file.

### Anti-UAV clip (the video file)

| Model | Input (w×h) | FPS | Frame time ms | P99 ms | Detect ms | Slowest stage (busy) | 120+ FPS | P99 < 16 ms | HOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 640×640 | 141.80–186.27 | 5.37–7.05 | 6.77–9.47 | 2.41–2.45 | capture (99.5–99.7%) | yes | yes | 32.74 | 37.05 | 541 |
| yolo26s_640x640 | 640×640 | 149.54–185.11 | 5.40–6.69 | 6.99–8.86 | 4.15–4.27 | capture (99.5–99.7%) | yes | yes | 37.81 | 43.28 | 503 |
| yolo26m_640x640 | 640×640 | 105.43–106.05 | 9.43–9.49 | 11.17–11.31 | 9.41–9.47 | detect (99.1%) | no | yes | 38.72 | 44.70 | 493 |
| **yolo26n_1280x736** | **1280×736** | **124.82–126.18** | 7.92–8.01 | **9.17–10.45** | 4.14–4.15 | **prep (99.2–99.3%)** | **yes** | **yes** | **41.55** | **49.33** | 772 |
| yolo26s_1280x736 | 1280×736 | 108.89–110.06 | 9.09–9.18 | 10.99–11.08 | 9.02–9.16 | detect (98.6–99.1%) | no | yes | 44.86 | 52.30 | 707 |
| yolo26m_1280x736 | 1280×736 | 46.77–47.20 | 21.18–21.38 | 24.46–24.62 | 21.16–21.36 | detect (99.6%) | no | no | 46.90 | 54.77 | 700 |

**The wide 640 ranges come from one slow round.** n and s at 640 ran 141.80 and 149.54 FPS in round 1, and 186.27 and 185.11 in round 2. They were the session's first two runs, and capture was the slowest stage in both rounds (busy 99.5–99.7%), so the decoder ran slower at first. Earlier n-only runs in the same session gave 186.19–187.95 FPS (P99 6.66–6.90 ms) at 640 and 126.50–127.08 FPS (P99 9.20–9.29 ms) at 1280×736. The slow round still met both targets.

### MOT17-04 (JPEG files)

| Model | Input (w×h) | FPS | Frame time ms | P99 ms | Detect ms | Slowest stage (busy) | 120+ FPS | P99 < 16 ms |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 640×640 | 65.81–65.89 | 15.18–15.20 | 20.08–20.25 | 2.86–2.94 | capture (99.9%) | no | no |
| yolo26s_640x640 | 640×640 | 65.83–66.07 | 15.13–15.19 | 19.95–20.00 | 4.73–4.75 | capture (99.9%) | no | no |
| yolo26m_640x640 | 640×640 | 64.73–65.09 | 15.36–15.45 | 19.84–20.04 | 8.44–8.46 | capture (99.8%) | no | no |
| yolo26n_1280x736 | 1280×736 | 64.53–64.81 | 15.43–15.50 | 20.22–20.41 | 4.35–4.39 | capture (99.8%) | no | no |
| yolo26s_1280x736 | 1280×736 | 65.13–65.19 | 15.34–15.35 | 19.75–19.92 | 8.20–8.22 | capture (99.8%) | no | no |
| yolo26m_1280x736 | 1280×736 | 47.43–47.49 | 21.06–21.08 | 23.88–24.13 | 21.03–21.06 | detect (99.7%) | no | no |

Five of the six rows run at the JPEG decoder's pace. Model and input size only show up once detect is slower than decoding a JPEG (m at 1280×736).

## Sequential (for reference)

Every step adds up, so the busy shares here are shares of one frame, and none is near 100%.

| Model | Input | Anti-UAV FPS | P99 ms | Detect ms | MOT17-04 FPS | P99 ms |
|---|---|---|---|---|---|---|
| yolo26n | 640×640 | 111.95–112.68 | 12.23–12.73 | 2.47–2.53 | 50.62–51.18 | 22.61–23.37 |
| yolo26s_640x640 | 640×640 | 90.41–91.24 | 14.73–15.84 | 4.36–4.46 | 45.53–45.65 | 27.63–27.72 |
| yolo26m_640x640 | 640×640 | 66.57–66.81 | 19.50–19.75 | 8.41 | 38.51–38.88 | 30.54–30.59 |
| yolo26n_1280x736 | 1280×736 | 64.41–65.52 | 20.21–20.31 | 4.55–4.77 | 35.13–37.23 | 33.72–37.55 |
| yolo26s_1280x736 | 1280×736 | 51.47–51.57 | 21.99–22.11 | 8.26–8.31 | 27.81–31.70 | 41.27–45.87 |
| yolo26m_1280x736 | 1280×736 | 33.40–33.45 | 33.01–33.22 | 18.58–18.72 | 24.28–24.32 | 52.59–54.57 |

No row meets 120 FPS sequentially. n at 640 meets the P99 target on the Anti-UAV clip, and s at 640 meets it there too, by 0.16 ms in its slower round. Full tables with busy shares are in `model-size.json` (`speed`).

## Why accuracy moves: detector recall

All 7 videos, person class, score ≥ 0.25, the official rule, no tracker:

| Model | 640×640 recall / precision | 1280×736 recall / precision |
|---|---|---|
| n | 33.7% / 85.3% | 54.6% / 84.4% |
| s | 40.0% / 86.3% | 58.0% / 82.8% |
| m | 41.5% / 87.0% | 60.3% / 82.9% |

- **Input size is still the bigger lever.** n goes from 33.7% to 54.6% recall at 1280×736, a 20.9-point gain, as in ADR 0002.
- **But model size isn't nothing.** At 640, s adds 6.2 points and m 7.8; at 1280×736, s adds 3.3 and m 5.6. ADR 0002's two-video sample (MOT17-02 and 04) suggested bigger models at 640 "barely" helped; across all 7 videos, s at 640 is worth 5.07 HOTA over n.
- **ID switches go up with recall,** because more people are tracked. At 1280×736 there are 700–772 against 493–541 at 640. A raw count of switches isn't normalized by how many people are found, so read it next to HOTA and IDF1, which are.

## How much of a row-to-row difference is real

The same tracker on the same model, with detections from ONNX Runtime on the CPU (the existing `results/mot17/detections/` sets) instead of TensorRT, scored:

| Detections | n at 640 (HOTA / IDF1 / IDSW) | n at 1280×736 |
|---|---|---|
| ONNX Runtime (CPU) | 33.13 / 37.62 / 529 | 41.46 / 49.38 / 732 |
| TensorRT (this table) | 32.74 / 37.05 / 541 | 41.55 / 49.33 / 772 |

The runtimes agree within #16's tolerance (scores within 0.00032), yet HOTA moved by up to 0.39, IDF1 by up to 0.56 and ID switches by 12–40. The tracker's thresholds turn tiny score differences into different decisions. **Treat gaps under about 0.4 HOTA, 0.6 IDF1 or 40 ID switches as noise.** Every gap the recommendation rests on (n at 1280×736 against the other rows that meet the targets: +3.73 HOTA or more) is well outside that.

## Recommendation

**Ship YOLO26n at 1280×736** (`models/yolo26n_1280x736.onnx` and its `--noTF32` engine), in parallel mode with TensorRT.

- It's the most accurate row that meets both targets on the video file: +8.81 HOTA and +12.27 IDF1 over n at 640, and +3.73 HOTA / +6.05 IDF1 over s at 640, the other row that does.
- **The margin is thin:** 124.82–126.18 FPS against 120. A slower decoder, a busier CPU or `--show` (which #13 measured at 10.16 and 13.88 ms a frame of draw) would put it under. P99 has more room: 9.17–10.45 ms against 16.
- **Prep is the slowest stage** for this row on the Anti-UAV clip (busy 99.2–99.3%; capture 74.8–75.2%, detect 51.4–51.8%). **So #17 (CUDA frame prep) is a go**: it attacks exactly the step that sets this row's FPS. #15 put the ceiling with prep removed at 156.8–160.0 FPS, with detect next. On MOT17-04, capture (JPEG decode) is the slowest stage for every row but m at 1280×736, so CUDA prep won't change FPS there.
- **ADR 0002's split goes away:** the benchmark detection set (n at 1280×736) and the shipped detector would be the same model.

What it costs and what's left open:
- **The spec's targets were written for n at 640.** If this ships, the SPEC "Targets" line, `DEFAULT_MODEL`/`--model-size` and ADR 0002 need updating. That isn't done here, because the choice is the user's.
- **s at 1280×736 is +3.30 HOTA, but runs 108.89–110.06 FPS** against 124.82–126.18, bounded by detect, not prep. Only a faster detect gets it under the targets. An FP16 export is the obvious candidate (#16 noted TensorRT 11 needs an FP16 ONNX for that). It's untried, and it would need its own agreement check.
- **n at 640 stays the fastest** (186.27 FPS in its fast round, P99 6.77 ms) if the margin above ever matters more than accuracy.

## Caveats

- **One laptop, one Anti-UAV clip, two rounds each.** The first round ran slow for the first two models (see above). The FPS margin for the recommended row (4.01–5.15% over the target) is far smaller than the first-round slowdown seen here (141.80 against 186.27 FPS for n at 640), so it should be re-measured after any change upstream of detect.
- **MOT17 is the tuning set** (no held-out split), as everywhere in Tether. The accuracy columns compare detectors under the same tracker, which is fair, but the absolute numbers are on data the tracker settings were chosen on.
- **Accuracy comes from the TensorRT dumps** of the shipped runtime, so the n rows differ slightly from the ONNX Runtime numbers in `mot17-bytetrack.md` and `edge-rule.md` (table above).
- **Headless and on video files only.** `--show`, `--record` and the webcam weren't timed with any of these models.
- **The agreement check for the new engines covered only 100 frames** of MOT17-04 (3,932–10,470 detections), not both full videos as in #16.
- **Square 1280×1280 and other sizes (e.g. 960×544) weren't run.** The issue asked for n/s/m × {640, 1280}, and ADR 0002 already showed 1280×736 is 42% cheaper than the square on the CPU, for about 1 point less recall.

## Commands

```
python tools/export_model.py --model s --size 640          # and s 1280x736, m 640, m 1280x736
trtexec --onnx=models/yolo26s_640x640.onnx --saveEngine=models/yolo26s_640x640.engine --noTF32 --skipInference
tether-runtime-compare data/MOT17/train/MOT17-04-FRCNN/img1/%06d.jpg --model models/yolo26s_640x640.onnx --rounds 1 --max-frames 100
tether-pipeline data/MOT17/train/<SEQ>-FRCNN/img1/%06d.jpg --model models/<M>.onnx --runtime tensorrt --dump results/model-size/detections/<M>/<SEQ>-FRCNN.jsonl
tether-mot17 data/MOT17 --model models/<M>.onnx --work-dir results/model-size --name tether_<M>
python tools/detector_recall.py data/MOT17 results/model-size/detections/<M> --work-dir results/model-size
tether-model-compare VIDEO --models models/yolo26n.onnx models/yolo26s_640x640.onnx models/yolo26m_640x640.onnx \
    models/yolo26n_1280x736.onnx models/yolo26s_1280x736.onnx models/yolo26m_1280x736.onnx \
    --rounds 2 --pipeline parallel --accuracy-dir results/model-size/trackers --out results/model-size/final/<video>-parallel.json
```
