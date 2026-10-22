# Drone detection: the fine-tuned drone model on the leak-free val

Run 2026-10-22 for #43. This is **YOLO26n at 1280×736 fine-tuned on Anti-UAV with one class, "drone"** (`models/yolo26n_drone_1280x736.onnx`, [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md)), scored on the **session-disjoint val** (`splits/antiuav.json`, seed 0), the same val as the COCO "before" row in [drone-coco-baseline](drone-coco-baseline.md). The checkpoint and the threshold were both chosen on this val, so these are tuned numbers. Anti-UAV test and Seraphim weren't touched.

Raw results, all from `tether-drone-score` unless named:
- [`drone-model.json`](drone-model.json): the chosen model.
- [`drone-model-candidates.json`](drone-model-candidates.json): the pilot and the six checkpoints compared.
- [`drone-model-breakdown.json`](drone-model-breakdown.json): recall by session, from `tools/drone_breakdown.py`. Its `by_class` names class 0 by COCO's table ("person"), which doesn't apply to this model.
- [`drone-model-absent-detections.json`](drone-model-absent-detections.json): every detection on a drone-absent frame at the chosen threshold, with its distance in frames to the nearest labeled drone and whether it touches the frame's edge.

Training record:
- [`drone-model-dataset.json`](drone-model-dataset.json): the training set's stride, image width and counts, from `tether-drone-dataset`.
- [`drone-model-train-args.yaml`](drone-model-train-args.yaml): Ultralytics' full settings.
- [`drone-model-train-results.csv`](drone-model-train-results.csv) and [`drone-model-pilot-results.csv`](drone-model-pilot-results.csv): its per-epoch losses, val metrics and times for the full run and the pilot.

## Results

Model: `yolo26n_drone_1280x736`, epoch 19 of 20. Detections came from the C++ pipeline on TensorRT at a **0.01 floor**, the same runtime and floor as the COCO row it's compared with. AP50 is **0.995**, against **0.228** for the COCO model.

| Model | Threshold | Chosen by | Recall | Precision | F1 | False alarms / min (drone absent) |
|---|---|---|---|---|---|---|
| **drone, epoch 19** | **0.352** | best F1 on val | **0.992** (37,961 of 38,250) | **0.995** | **0.994** | 46.8 (48 in 1.03 min; see caveats) |
| drone, epoch 19 | 0.25 | the tracker's min score | 0.994 (38,005 of 38,250) | 0.992 | 0.993 | 62.3 (64 in 1.03 min) |
| COCO, class-agnostic | 0.066 | best F1 on val | 0.397 (15,194 of 38,250) | 0.270 | 0.321 | 137.3 (141 in 1.03 min) |
| COCO, class-agnostic | 0.25 | the tracker's min score | 0.142 (5,414 of 38,250) | 0.410 | 0.210 | 7.8 (8 in 1.03 min) |

Recall by labeled box size, at source resolution (area: tiny < 16², small < 32², medium < 96², large ≥ 96² px):

| Size | Drones | Drone model at 0.352 | Drone model at 0.25 | COCO at 0.25 |
|---|---|---|---|---|
| tiny | 8 | 0 of 8 | 0 of 8 | 0 of 8 |
| small | 12 | 3 of 12 | 3 of 12 | 4 of 12 |
| medium | 20,530 | 0.989 | 0.990 | 0.127 |
| large | 17,700 | 0.998 | 0.998 | 0.158 |

Recall by val session, over the frames where the drone is labeled:

| Session | Drone frames | Drone model at 0.352 | Drone model at 0.25 | COCO at 0.25 |
|---|---|---|---|---|
| 20190925_131530 | 6,889 | 0.997 | 0.998 | 0.011 |
| 20190925_133630 | 9,443 | 0.986 | 0.987 | 0.071 |
| 20190925_205804 | 6,927 | 0.994 | 0.995 | 0.039 |
| 20190926_142435 | 8,036 | 0.996 | 0.997 | 0.219 |
| 20190926_143632 | 6,955 | 0.992 | 0.994 | 0.378 |

## Choosing the checkpoint

Every epoch's weights were kept. The rule was set before the full run:
1. Shortlist the three epochs with the best Ultralytics val mAP50, plus the last.
2. Score each with `tether-drone-score` on every val frame.
3. Ship the one with the highest AP50.

Ultralytics' mAP50 is computed on the stride-10 val images at a 1280 square. It flattened at 0.994–0.995 from epoch 7 on, so it couldn't separate the late epochs. The first shortlist (epochs 13, 18, 20) was cut through a tie: 18 and 20 had 0.99487, and 16, 17 and 19 had 0.99486. Code review pointed this out, so the three tied epochs were scored too. The full-val AP50 is what chose.

| Checkpoint | Ultralytics val mAP50 | AP50 (`tether-drone-score`, full val) | Best-F1 threshold | F1 there | Recall / precision at 0.25 |
|---|---|---|---|---|---|
| pilot (1 epoch, separate run) | 0.97723 | 0.94590 | 0.236 | 0.896 | 0.876 / 0.915 |
| epoch 13 | 0.99488 | 0.99386 | 0.385 | 0.990 | 0.990 / 0.988 |
| epoch 16 | 0.99486 | 0.99437 | 0.275 | 0.993 | 0.992 / 0.993 |
| epoch 17 | 0.99486 | 0.99480 | 0.416 | 0.993 | 0.993 / 0.990 |
| epoch 18 | 0.99487 | 0.99485 | 0.339 | 0.994 | 0.994 / 0.992 |
| **epoch 19** | 0.99486 | **0.99503** | 0.352 | 0.994 | 0.994 / 0.992 |
| epoch 20 (last) | 0.99487 | 0.99495 | 0.347 | 0.994 | 0.994 / 0.992 |

Epoch 19 wins by 0.00008 AP50 over epoch 20. Epochs 17–20 are within 0.0003 of each other, far inside what five val sessions can resolve. Read it as "the late epochs are tied, and the rule picked one", not as epoch 19 being better. Epoch 19 is the 19th epoch, saved as `weights/epoch18.pt`.

## Training

- **Data:** `tether-drone-dataset` at stride 10 from the split's train clips gave 17,227 images. 997 of them (5.8%) have the drone absent, against 5.7% of all train frames. The 3,953 val images (stride 10, 3.1% absent) are only for Ultralytics' per-epoch check. Images are stored at 1280×720.
- **Settings** (`tools/train_drone.py`, `train_args`): starts from the COCO-pretrained `yolo26n.pt`, with 20 epochs, batch 8, imgsz 1280, seed 0, deterministic mode and `save_period=1`. Everything else is Ultralytics 8.4.181's default. The `auto` optimizer chose AdamW at lr 0.002. Mosaic is on and closed for the last 10 epochs, and the rest of the augmentation is default. PyTorch 2.14.1+cu130.
- **Time** on the RTX 5070 Laptop GPU (8 GB, 5.73 GB used):
  - The 1-epoch pilot took 717.6 s for training plus val, 14m34s with start-up. It had mosaic off, since its one epoch fell inside the last 10.
  - The full run took **3.39 hours** for 20 epochs (12,198 s, about 610 s per epoch).
- **Export:** `tools/train_drone.py export`, with the same settings as the shipped COCO export (`tools/export_model.py`): 1280×736, batch 1, the NMS-free head, opset 18. The class name `{0: 'drone'}` is in the ONNX metadata. Engines were built with `trtexec --noTF32 --skipInference` (TensorRT 11.4.0.106).

## What it means

- **Fine-tuning takes the detector from finding one drone in seven to nearly all of them.** At the tracker's 0.25, recall goes from 14.2% to 99.4% and precision from 41.0% to 99.2%. AP50 goes from 0.228 to 0.995 at the same floor and runtime.
- **One epoch already got most of it.** The pilot reached AP50 0.946, and the next epochs added the rest.
- **The session spread is gone.** The COCO model's recall at 0.25 ranged from 1.1% to 37.8% between sessions. The drone model's ranges from 98.7% to 99.8%. These are the same camera, site and two days the model trained on, though: see the caveats.
- **The threshold barely matters on val.** At 0.25 against the best-F1 cut of 0.352, recall is 0.1 points higher and precision 0.3 lower. Tracking at the current 0.25 (#44) loses almost nothing against the tuned cut.

## Caveats

- **Tuned on this val.** The checkpoint and the threshold were both picked here. Both choices were between options this close together that the bias is small, but the number for the claim is test (#48), run once.
- **One site, one camera, two days.** Val sessions are new flights, but every Anti-UAV clip comes from the same pan-tilt turret on 2019-09-25 and 26 (ADR 0004). 99% here says the model works on new flights at that site. It doesn't say it works anywhere else; that's what the Seraphim transfer test is for (#48).
- **Most "false alarms" are the drone at the edge of the view.** At 0.352, 48 detections sit on frames where the drone is labeled absent, far more than the COCO model's 8 at 0.25.
  - 42 of the 48 come within 10 frames of a labeled drone, 33 touch the frame's edge, and 47 do one or the other.
  - The last one, frame 260 of `20190926_142435_1_8`, ends 6 px above the bottom edge, 18 frames from a labeled drone.
  - Looked at by eye, frames 260, 263 and 309 of that clip show a dark shape at the bottom edge (260, 263) and part of a drone at the left edge (309), all labeled absent.

  So the rate measures mostly how the labels treat a drone half out of view, not alarms on empty sky. It also rests on 1.03 minutes of drone-absent footage. A fair empty-sky rate needs frames well away from any labeled drone, which this val has few of.
- **Small and tiny drones are too few to judge.** There are 20 of them in val (8 tiny, 12 small), and the drone model finds 3. ADR 0004 measured that at 1280×736 no labeled drone is under 8 px on its short side, so this tail is thin by construction.
- **Run-to-run variation wasn't measured.** It's one training run with seed 0. Deterministic mode was on, but cuDNN on a different GPU or driver may still not reproduce it bit for bit.

## Reproduce

```
pip install -e .[export]   # plus a CUDA build of PyTorch, e.g. torch==2.14.1+cu130 from download.pytorch.org/whl/cu130
tether-drone-dataset data/Anti-UAV300 data/antiuav-drone-stride10      # about 6.5 minutes
cd models    # Ultralytics' AMP check downloads yolo26n.pt into ./weights, so run from a folder out of git
python ../tools/train_drone.py train <abs>/data/antiuav-drone-stride10/data.yaml --epochs 20 --name full \
    --pretrained <abs>/models/yolo26n.pt --project <abs>/models/drone-runs
cd ..
python tools/train_drone.py export models/drone-runs/full/weights/epoch18.pt models/yolo26n_drone_1280x736.onnx
trtexec --onnx=models/yolo26n_drone_1280x736.onnx --saveEngine=models/yolo26n_drone_1280x736.engine --noTF32 --skipInference
tether-antiuav-detect data/Anti-UAV300 --model models/yolo26n_drone_1280x736.onnx --split val --det-min-score 0.01 \
    --work-dir results/antiuav-trt --detector "cpp/build-trt/Release/tether-pipeline.exe --runtime tensorrt --prep cuda"
tether-drone-score data/Anti-UAV300 results/antiuav-trt/detections/yolo26n_drone_1280x736/val --split val \
    --out docs/benchmarks/drone-model.json
```

`epochN.pt` is 0-based, so `epoch18.pt` is the 19th epoch, row 19 of the results CSV. Each candidate was exported and dumped under its own name (`yolo26n_drone_1280x736_e13`, `_e16` … `_e20`, `_pilot`, numbered 1-based), so they couldn't collide. The shipped file is byte-identical to `_e19`, and `drone-model.json` was scored from the `_e19` dump.
