# The edge rule: not handing a departed object's ID to a newcomer

Run 2026-10-19 for #37. Same setup as [antiuav-duplicates.md](antiuav-duplicates.md): Anti-UAV **val** (67 clips, all 1920×1080), seed 0. MOT17 is the 7 train videos with the dumped YOLO26n detections, at 1280×736 and 640. "Before" is `main` after #36 (merge 10 + 1, cap 0.5).

**Result: shelved, not merged.** The rule fixes the scripted walk-out/walk-in swap, but on the real data it costs more than it saves. Anti-UAV val loses about 1 point of ID kept at both deletion rates. MOT17-04 at 1280×736 regresses (two correct re-matches blocked). And on real footage the swap it targets turned out to be rare. The code and its tests are kept on the branch `issue-37-edge-rule-experiment`.

## The rule

A track whose last observed box touched the frame edge most likely left the scene, so neither the search gate nor the duplicate merge hands its ID to anyone else. IoU matching still applies, so an object standing at the edge keeps its ID through a missed frame. The tracker was given the frame size through a `frame_size` option (`tether-track --frame-size W H`; the harnesses filled it in per video).

### Picking "touching the edge"

The margin is a share of the box's **height**, because a box clipped at the edge loses width but keeps its height. To pick it, we took MOT17 ground-truth people whose box runs past an edge and found the detection matching them (IoU ≥ 0.5, score ≥ 0.25). Then we measured how far inside the frame that detection's edge sits, in detection heights (scratch, `results/i37/edge_slack.py` → `edge_slack.txt`):

| Edge | 640: p90 | 640: share within 0.05 | 1280×736: p90 | 1280×736: share within 0.05 |
|---|---|---|---|---|
| Left | 0.006 | 98.8% (n = 735) | 0.007 | 98.4% (n = 767) |
| Top | 0.008 | 99.6% (n = 1032) | 0.011 | 99.5% (n = 3515) |
| Right | 0.005 | 99.2% (n = 780) | 0.004 | 99.2% (n = 826) |
| Bottom | 0.017 | 97.5% (n = 2956) | 0.071 | 79.0% (n = 2897) |

**0.05 box heights** was chosen to cover 98–99% of the left, right and top cases. The bottom is looser at 1280×736, and the rule wasn't tuned for it separately. In the real webcam clip (below), a 926 px person cut off by the frame had their box edge up to 15 px (1.6%) inside the frame.

## Scripted scene

The scene from #36: 1920×1080, 30 fps. A person about 800 px tall walks out the left edge, and after a gap a same-size person walks in at the right edge. Before, the newcomer got the old ID for every gap from 0 to 29 frames. With the rule, the newcomer got a new ID at every gap tested (0, 1, 5, 15 and 29 frames). That held through the gate alone (merge off, gaps 5–29) and through the merge alone (gate off, gap 0). An object lost mid-frame was still re-matched by the gate and merged as before.

## How common is the risky pattern? (MOT17 ground truth)

The pattern: a person (A) disappears, and within 30 frames a different, same-size person (B) appears within the gate or merge reach of A's prediction, with no one else close enough for the cap to bind. Pedestrians only. A's prediction is constant velocity from its last 5 frames. The reaches are the tracker defaults, checked on B's second frame for the gate and on its third frame for the merge. Scratch: `results/i37/risky_mot17.py` → `risky_mot17.txt`.

| "Disappears" means | Disappearances | ...at the edge | Risky | ...of which A was at the edge |
|---|---|---|---|---|
| A's ground-truth track ends | 408 | 220 | 40 | 3 |
| A's visible run ends (visibility ≥ 0.25 then below or gone) | 1074 | 259 | 194 | 23 |

Per video, by visible run: MOT17-02 29 risky (0 at the edge), -04 5 (3), -05 56 (14), -09 15 (3), -10 16 (2), -11 12 (0), -13 61 (1).

**Most of the risk on MOT17 is mid-frame**: 88.1% of the risky cases by visible run, 92.5% by track end. Edge exits are common (about half of all track ends), but a newcomer rarely shows up within reach of one. The rule can only touch the edge cases. This is a ground-truth proxy, not a count of tracker swaps.

## Anti-UAV val

What share of the drone's losses happen at the frame edge? Scratch: `results/i37/au_edge_losses.py` → `au_edge_losses.txt`.

- Ground truth, labeled → unlabeled: 6 of 25 (24%) at the edge.
- The tracker at the #36 defaults, tracked → searching, with the last box at the edge:

| Condition | Losses | At the edge |
|---|---|---|
| 0% deleted | 1506 | 9 (0.6%) |
| 20% deleted | 9602 | 36 (0.4%) |
| 0.25 s gaps | 2530 | 11 (0.4%) |
| 0.5 s gaps | 2267 | 11 (0.5%) |
| 1 s gaps | 1855 | 11 (0.6%) |

Before and after, committed code with the rule on:

| | ID kept 0% | ID kept 20% | 0.25 s gaps | 0.5 s gaps | 1 s gaps |
|---|---|---|---|---|---|
| Before (#36) | 88.4% | 85.3% | 92.0% | 88.0% | 88.8% |
| Edge rule | 87.3% | 84.3% | 91.8% | 87.6% | 88.4% |

The gap rows are over the same held gaps (1206 / 1065 / 874), and the prediction error didn't change. **The loss is bigger than the 0.4–0.6% edge share suggests.** Our guess, not checked: ID kept counts only the clip's most common ID, so one blocked re-match early in a clip can cost every frame after it.

## MOT17

HOTA / IDF1 / ID switches, before → edge rule:

| Video | 1280×736 | 640 |
|---|---|---|
| MOT17-02 | 27.49 / 32.05 / 136 → same | 23.92 / 25.88 / 57 → same |
| MOT17-04 | **45.62 / 54.16 / 177 → 45.55 / 54.06 / 178** | 31.51 / 33.97 / 64 → same |
| MOT17-05 | 42.66 / 58.39 / 76 → same | 42.33 / 55.81 / 76 → 42.34 / 55.84 / 74 |
| MOT17-09 | 44.59 / 52.69 / 67 → 44.65 / 52.72 / 67 | 39.01 / 45.11 / 64 → same |
| MOT17-10 | 35.85 / 44.87 / 134 → 35.87 / 44.88 / 134 | 30.84 / 38.95 / 152 → same |
| MOT17-11 | 49.09 / 52.47 / 36 → same | 48.10 / 52.84 / 35 → 48.11 / 52.84 / 35 |
| MOT17-13 | 37.51 / 49.00 / 106 → 37.79 / 49.84 / 107 | 27.82 / 34.51 / 81 → 27.83 / 34.52 / 81 |
| Combined | 41.46 / 49.38 / 732 → 41.46 / 49.42 / 734 | 33.13 / 37.62 / 529 → 33.13 / 37.62 / 527 |

**MOT17-04 at 1280×736 regresses on all three metrics.** That fails #37's per-video check.

To see what the rule blocks, we logged, with the rule off, every gate match or merge whose track's last box touched the edge (scratch, `results/i37/blocked.py`, `outward.py`). There are few: 15 at 1280×736 and 8 at 640, across all 7 videos, all gate matches and no merges.

- On MOT17-04 both blocked matches were **correct**: the same ground-truth person on both ends. These were people standing partly cut off at the top edge (visibility 0.29–0.43) who were missed for 6 and 20 frames. They weren't leaving.
- One of MOT17-13's two blocked matches was also correct.
- Most of the rest are tracks whose last box doesn't match a scored pedestrian at IoU 0.3. That's unknown either way: ground-truth boxes run past the frame, so a clipped detection may simply overlap them poorly.

### A variant: also require moving out

Narrowing the rule to tracks whose Kalman center velocity points out through the edge they touch (scratch, `results/i37/variant_cli.py`), MOT17 only, not run on Anti-UAV:

| 1280×736 | MOT17-04 | MOT17-09 | MOT17-13 | Combined |
|---|---|---|---|---|
| Before (#36) | 45.62 / 54.16 / 177 | 44.59 / 52.69 / 67 | 37.51 / 49.00 / 106 | 41.46 / 49.38 / 732 |
| Plain rule | 45.55 / 54.06 / 178 | 44.65 / 52.72 / 67 | 37.79 / 49.84 / 107 | 41.46 / 49.42 / 734 |
| Outward velocity > 0 | 45.55 / 54.06 / 178 | 44.63 / 52.70 / 67 | 37.79 / 49.84 / 107 | 41.46 / 49.42 / 734 |
| Outward velocity > 0.01 heights/frame | 45.62 / 54.16 / 177 | 44.59 / 52.69 / 67 | 37.79 / 49.84 / 107 | 41.49 / 49.46 / 733 |

640 is identical for all three versions of the rule. The two MOT17-04 tracks had outward velocities of +0.007 and −0.003 heights/frame, which is noise. So 0.01 is a threshold read off two samples of the check set. It also lets slow walk-outs through: in the scripted scene at 5 and 10 px/frame (0.006 and 0.0125 heights), the gap-0 newcomer still took the old ID. We didn't adopt it.

## Real footage

`results/dup36/sparse.mp4` is 139 frames at 1920×1080. One person walks out the right edge and another walks in, both close to the camera. It was run through the tracker the demo uses, on dumped detections, without and with the rule:

- **640 (the demo's model):** no change. The person's last box before they disappeared was a 245 px fragment and the newcomer was about 990 px tall, so **the size check already refused the swap**.
- **1280×736:** the detector sees only a leg fragment (about 300 px) as each person crosses the edge. Before, the newcomer's legs took the departed person's fragment ID. With the rule, the newcomer's legs got a new ID. In both runs the newcomer's ID then breaks once more when their box grows from legs to full body, because the size check refuses that.

## Caveats

- One real clip and one scripted scene. Neither says how often the swap happens on real footage.
- The MOT17 risky-pattern count runs on ground truth, with a constant-velocity stand-in for the tracker's prediction. It's a proxy, not a count of tracker swaps.
- The reason behind the Anti-UAV loss wasn't checked.
- What the rule can't do in principle: the tracker only knows boxes and motion. Once someone leaves, the same person walking back in looks exactly like a newcomer. The rule gives both a new ID; before, both got the old one. Telling them apart needs appearance re-identification, which #37 put out of scope.

## Reproduce

Check out `issue-37-edge-rule-experiment`, then:

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val/i37 --name edge
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name i37_edge_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n.onnx --name i37_edge_yolo26n
```

On that branch the harnesses pass `--frame-size {width} {height}` by default. Without `--frame-size`, `tether-track` behaves as on `main`. The scratch scripts named above are in `results/i37/` and aren't committed.
