# Merging duplicate tracks of one object

Run 2026-10-19 for #36. Same setup as [antiuav-search-gate-cap.md](antiuav-search-gate-cap.md): Anti-UAV **val** (67 clips, 20 fps), seed 0. MOT17 is the 7 train videos with the dumped YOLO26n detections, at 1280×736 and 640. #31 (tentative tracks surviving a miss) is not on `main`, so every number here has a tentative track dropped on its first miss.

**Result: a duplicate is merged into the older track when it's confirmed, with its own reach of 10 + 1 box sizes per missed frame, capped and size-checked like the gate, on by default.** ID kept at 20% deletion goes from 59.9% to 85.3%. MOT17 doesn't move at either input size: every video is within 0.01 HOTA of its committed score.

## Measure first

At the #35 defaults, through scratch scripts that track in-process (`results/dup36/`, not committed). On Anti-UAV every detection is the drone, so every confirmed track is a track of it.

| | 0% deleted | 20% deleted |
|---|---|---|
| ID kept | 60.2% | 59.9% |
| Scored frames with two or more confirmed tracks | 25.9% | 23.8% |
| ...with two predicted within 3 box sizes of the drone | 5.7% | 3.9% |
| Holder changes | 913 | 785 |
| ...to a track born while the old holder was searching | 863 (94.5%) | 724 (92.2%) |
| ...back to an older track | 35 | 46 |
| ...to a track born after the old holder was removed | 15 | 15 |
| ID kept if every such track had taken the oldest live ID (oracle) | 91.1% | 89.2% |

In every one of the 863 / 724 changes the old holder was still alive. So the share is not small, and duplicates are nearly all of what's left of ID kept.

## Why the issue's two approaches don't work as written

Both "merge when the old prediction sits on the new track" and "block confirmation while the old gate covers it" need the duplicate to be born near the old track's prediction. At each birth of a track that went on to be confirmed while an older track searched, the old prediction was this far from it (in the old track's box sizes):

| Percentile | 10th | 25th | 50th | 75th | 90th |
|---|---|---|---|---|---|
| 0% deleted | 1.1 | 2.9 | 5.4 | 8.6 | 15.8 |
| 20% deleted | 1.4 | 3.3 | 5.6 | 8.9 | 15.8 |

67–75% were outside the uncapped gate on distance. Under 1.1% failed the size check. The rest (24–33%) were inside the uncapped gate but blocked by the cap, because another duplicate sat nearby. Static ceilings (the oracle above, but only for tracks born within the given reach of an older searching track of the same size):

| Reach | none | the gate | 3 | 5 | 10 | 20 | unlimited |
|---|---|---|---|---|---|---|---|
| ID kept 0% | 60.2% | 66.0% | 70.7% | 79.1% | 89.8% | 90.1% | 90.4% |
| ID kept 20% | 59.9% | 61.5% | 63.4% | 69.0% | 84.2% | 87.1% | 88.2% |

A prototype merge at confirmation (scratch, `results/dup36/proto.py`) at the gate's own reach and cap made **0 merges** on val: every in-gate duplicate is already cap-blocked. So the merge gets its own, wider reach, and the cap is what keeps it safe in a crowd.

## Tuning

Prototype, merge at confirmation, cap 0.5, size check on:

| Reach r0 + k/frame | Cap | ID kept 0% | ID kept 20% | 1 s gaps | Merges at 20% | MOT17 merges (1280×736 / 640) |
|---|---|---|---|---|---|---|
| off (#35 default) | | 60.2% | 59.9% | 86.6% | | |
| 1.5 + 0.25 | 0.5 | 60.2% | 59.9% | 86.6% | 0 | |
| 1.5 + 0.25 | none | 60.7% | 60.5% | 86.8% | 65 | |
| 3 + 0.25 | 0.5 | 65.3% | 64.2% | 87.9% | 227 | 3 / 3 |
| 5 + 0.5 | 0.5 | 76.0% | 73.1% | 88.6% | 588 | |
| 10 + 0.5 | 0.5 | 88.4% | 81.2% | 88.8% | 700 | 3 / 3 |
| 10 + 0.5 | none | 88.4% | 81.2% | 88.8% | 701 | 854 / 539 |
| **10 + 1** | **0.5** | **88.4%** | **85.3%** | **88.8%** | 717 | 3 / 3 |
| 20 + 0.5 | 0.5 | 88.4% | 86.2% | 88.8% | 724 | |
| 20 + 1 | 0.5 | 88.4% | 86.7% | 88.8% | 725 | |
| unlimited | 0.5 | 88.4% | 87.4% | 88.8% | 728 | |

- **The cap does all the safety work on MOT17.** With it, 3 merges at any reach from 3 to 10; without it, 854 at 1280×736. MOT17 scores were identical for reaches 3 + 0.25, 5 + 0.5 and 10 + 0.5 at both sizes.
- **Past 10 box sizes the gain levels off** (0% deletion stops at 88.4%; 20% creeps up 2 points by unlimited). 10 + 1 was chosen as the smallest reach near the top, not the best on val: each extra box size of reach is more room for a wrong merge in a sparse scene, which no benchmark here checks.

## Results at the new defaults

Committed code, plain `tether-track`, all conditions:

| | ID kept 0% | ID kept 20% | 0.25 s gaps | 0.5 s gaps | 1 s gaps |
|---|---|---|---|---|---|
| #35 default | 60.2% | 59.9% | 91.8% | 86.9% | 86.6% |
| **merge 10 + 1** | **88.4%** | **85.3%** | **92.0%** | **88.0%** | **88.8%** |
| SPEC target | — | at least 95% | — | — | at least 80% |

1 s gaps are recovery over 874 held gaps of 928 (876 before).

| | Holder changes 0% / 20% | Frames with 2+ confirmed tracks 0% / 20% | Cap binds, Anti-UAV 0% / 20% | Cap binds, MOT17 1280×736 / 640 |
|---|---|---|---|---|
| #35 default | 913 / 785 | 25.9% / 23.8% | 34.7% / 30.2% | 96.5% / 96.2% |
| merge 10 + 1 | 29 / 37 | 0.5% / 0.8% | 0.6% / 0.6% | 96.5% / 96.1% |

Binds use #35's definition (of gate checks with leftover detections, the share where the cap is below the full gate), recounted here at the current defaults. **The bind rate on Anti-UAV falls from about a third to 0.6%,** so the duplicates were what made the cap bind there, as #35 guessed. On MOT17 it doesn't change: the cap still binds almost every time, which is what keeps the wide reach out of the crowd.

**MOT17**, HOTA / IDF1 / ID switches, against the committed #35 numbers:

| | 1280×736 | 640 |
|---|---|---|
| #35 default | 41.46 / 49.38 / 731 | 33.13 / 37.62 / 529 |
| merge 10 + 1 | 41.46 / 49.38 / 732 | 33.13 / 37.62 / 529 |

Per video, committed #35 → merge 10 + 1:

| Video | Size | HOTA | IDF1 | ID switches |
|---|---|---|---|---|
| MOT17-02 | 1280×736 | 27.49 → 27.49 | 32.05 → 32.05 | 136 → 136 |
| MOT17-04 | 1280×736 | 45.62 → 45.62 | 54.16 → 54.16 | 177 → 177 |
| MOT17-05 | 1280×736 | 42.65 → 42.66 | 58.29 → 58.39 | 75 → 76 |
| MOT17-09 | 1280×736 | 44.59 → 44.59 | 52.69 → 52.69 | 67 → 67 |
| MOT17-10 | 1280×736 | 35.85 → 35.85 | 44.87 → 44.87 | 134 → 134 |
| MOT17-11 | 1280×736 | 49.09 → 49.09 | 52.47 → 52.47 | 36 → 36 |
| MOT17-13 | 1280×736 | 37.51 → 37.51 | 49.00 → 49.00 | 106 → 106 |
| MOT17-02 | 640 | 23.92 → 23.92 | 25.88 → 25.88 | 57 → 57 |
| MOT17-04 | 640 | 31.51 → 31.51 | 33.97 → 33.97 | 64 → 64 |
| MOT17-05 | 640 | 42.33 → 42.33 | 55.81 → 55.81 | 76 → 76 |
| MOT17-09 | 640 | 39.01 → 39.01 | 45.11 → 45.11 | 64 → 64 |
| MOT17-10 | 640 | 30.84 → 30.84 | 38.95 → 38.95 | 152 → 152 |
| MOT17-11 | 640 | 48.10 → 48.11 | 52.84 → 52.84 | 35 → 35 |
| MOT17-13 | 640 | 27.82 → 27.82 | 34.51 → 34.51 | 81 → 81 |

The defaults make 3 merges on all of MOT17 at each size. The one change is MOT17-05 at 1280×736: +1 ID switch (75 → 76), with IDF1 +0.10 (58.29 → 58.39) and HOTA +0.01. Our guess, not checked on that video: one of the 3 merges handed a person back to their old ID, which CLEAR counts as a switch while IDF1 rewards it. **We don't count it as a regression** under the issue's "off by default if MOT17 regresses" rule, because HOTA and IDF1, the two metrics that score the whole identity, are flat or up on every video. One extra ID switch out of 731 is the only movement. If you read the rule strictly, per metric, this is the cell that would flip it.

**Distance to the SPEC targets: ID kept at 20% deletion 85.3% vs 95%, 9.7 points short (was 35.1). 1 s recovery 88.8% vs 80%, met.** With 29–37 holder changes left, what's left is mostly frames where no track holds the drone, not frames held by the wrong ID.

## Caveats

- **Sparse scenes aren't tested.** MOT17 is crowded, so the cap binds, and on Anti-UAV a merge can't be wrong. In between, a lone person who leaves can hand their ID to a same-size newcomer up to 10 + 1 per missed frame box sizes away (up to about 40 heights after 30 frames), if the newcomer is confirmed before the old track is removed.
- **The older ID survives because of #31's current state.** Nothing compares ages. A tentative track is dropped on its first miss, so it is always younger than a track that was confirmed and then missed. If #31 lets tentative tracks survive a miss, that stops being guaranteed, and the merge should check ages.
- **The reach is in the old track's predicted box size**, as the gate is. A coasted prediction's size drifts (see the MOT17-09 diagnosis in antiuav-search-gate-cap.md), and at 10 + 1 per frame that drift moves the reach more than it moves the gate. The size check compares against the last observed box, so at least the new box must match the real size.
- The reach was tuned on val; test stays unseen.

## Reproduce

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val/dup36 --name default
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name i36_default_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n.onnx --name i36_default_yolo26n
```

The old behavior is `tether-track --merge-reach 0 --merge-reach-growth 0`. Duplicate counts, ceilings, prototype grid and bind counts: `results/dup36/measure.py`, `birth.py`, `ceiling.py`, `eval_au.py`, `binds.py` (scratch, not committed).
