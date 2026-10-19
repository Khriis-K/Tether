# Capping the search gate by the distance to the nearest other track

Run 2026-10-19 for #35. Same setup as [antiuav-search-gate.md](antiuav-search-gate.md) and [antiuav-coast-damping.md](antiuav-coast-damping.md): Anti-UAV **val** (67 clips, 20 fps), seed 0. MOT17 is the 7 train videos with the dumped YOLO26n detections, at 1280×736 and 640.

The change: a searching track's gate (`search_gate + search_gate_growth · frames_since_seen` box sizes, from #29) is capped at `search_gate_cap` times the center distance from its prediction to the nearest other *confirmed* track's prediction, in the searching track's box sizes. Alone, the cap never binds; in a crowd it keeps the gate inside the space that's this track's own. `tether-track --search-gate-cap C`; `inf` turns it off.

Tentative tracks don't count toward the cap. The failure the gate fixes on Anti-UAV is a track missing one jump: the detection starts a tentative track there, and the gate has to take the detection back from it. Measuring the cap to that tentative track would shut the gate exactly when it's needed.

**Result: the gate is now on by default, at 1.5 + 0.25/frame with cap 0.5, and damping stays off (1.0).** It's the only setting tried that leaves MOT17 at or above the committed numbers at both input sizes, while keeping almost all of the #29 gain on Anti-UAV.

## Which branch of the issue applied

From #30's numbers ([comment on #35](https://github.com/Khriis-K/Tether/issues/35)): with the gate on, damping 0.9 took MOT17 at 1280×736 from 41.29 / 49.04 / 748 (HOTA / IDF1 / ID switches) to 36.50 / 41.95 / 1318. The gate on top of damping didn't hold MOT17, so the cap was built. The issue planned it on top of damping; the numbers below say damping should stay off with the cap (see "Damping").

## Results

Gate 1.5 + 0.25 in every row but the first, the Anti-UAV val best from #29 and #30.

| Damping | Cap | ID kept 0% | ID kept 20% | 1 s gaps | MOT17 1280×736 | MOT17 640 |
|---|---|---|---|---|---|---|
| committed (no gate) | | 48.3% | 30.5% | 31.3% | 41.29 / 49.04 / 748 | 32.82 / 36.95 / 557 |
| 1.0 | none (#29) | 60.8% | 61.1% | 88.4% | 38.88 / 44.92 / 1261 | |
| **1.0** | **0.5** | **60.2%** | **60.5%** | **87.1%** | **41.33 / 49.23 / 733** | **32.94 / 37.30 / 531** |
| 1.0 | 0.75 | 60.2% | 60.4% | 87.4% | 41.24 / 49.13 / 739 | 32.98 / 37.46 / 532 |
| 0.9 | 0.5 | 61.0% | 61.8% | 91.9% | 41.29 / 49.28 / 734 | 32.36 / 36.84 / 564 |
| 0.8 | none (#30) | 63.2% | 63.7% | 94.3% | 35.07 / 39.79 / 1305 | |
| 0.8 | 0.3 | 61.8% | 62.4% | 91.2% | 41.64 / 49.27 / 762 | 32.99 / 37.33 / 581 |
| 0.8 | 0.5 | 61.8% | 62.5% | 93.5% | 41.80 / 49.75 / 743 | 32.92 / 37.35 / 573 |
| 0.8 | 0.75 | 61.7% | 62.2% | 94.4% | 41.87 / 49.78 / 742 | 33.05 / 37.42 / 575 |
| 0.8 | 1.0 | 61.7% | 62.8% | 94.4% | 41.17 / 48.92 / 823 | |
| 0 | none (#30) | 63.5% | 64.0% | 94.7% | 33.02 / 36.83 / 1328 | |
| 0 | 0.5 | 62.1% | 62.5% | 94.3% | 41.50 / 49.75 / 836 | 32.01 / 36.50 / 580 |
| 0 | 1.0 | 62.0% | 62.0% | 94.7% | | |
| SPEC target | | — | at least 95% | at least 80% | | |

Blank cells weren't run. 1 s gaps are recovery over 874–880 held gaps of 928. MOT17 is HOTA / IDF1 / ID switches. The uncapped MOT17 cells at damping 0.8 and 0 were run for this issue (#30 measured only 0.9 with the gate).

- **The cap fixes the MOT17 cost of the gate.** Undamped, it takes MOT17 at 1280×736 from 38.88 / 44.92 / 1261 back to 41.33 / 49.23 / 733, and at 640 to 32.94 / 37.30 / 531, all at or above the committed numbers. The margins are thin at 1280×736 (+0.04 HOTA, +0.19 IDF1, −15 ID switches) and clearer at 640 (+0.12, +0.35, −26). Per video it isn't uniform: MOT17-09 loses 1.8 HOTA at 1280×736 (44.7 → 42.9) and 2.1 at 640 (39.0 → 36.9), and MOT17-05 loses 0.8 at 1280×736, while 02, 04, 10 and 13 gain (per-video HOTA in [mot17-bytetrack.md](mot17-bytetrack.md)). Not investigated.
- **It costs Anti-UAV little.** Undamped, ID kept goes from 61.1% to 60.5% at 20% deletion, and 1 s recovery from 88.4% to 87.1%. Against the committed tracker that's still 30.5% → 60.5% and 31.3% → 87.1%.
- **The fraction barely matters on Anti-UAV** (0.3–1.0 all within about 1 point of ID kept), and on MOT17 1.0 is too loose (823 ID switches at damping 0.8). Undamped, 0.5 and 0.75 are close; 0.75 drops HOTA at 1280×736 just below committed (41.24), so the default is 0.5, which is also the issue's untuned starting value.
- **The defaults, end to end** (plain `tether-track`, all drop rates and gap lengths): ID kept 60.2% / 60.5%, and recovery of 91.9% of 0.25 s gaps, 87.2% of 0.5 s and 87.1% of 1 s (committed: 72.7% / 54.4% / 31.3%). MOT17 matches the table exactly at both sizes.
- **Distance to the SPEC targets (default row): 1 s recovery 87.1% vs 80%, met. ID kept at 20% deletion 60.5% vs 95%, still 34.5 points short.**

### Damping

With the cap, damping still helps Anti-UAV (0.8: ID kept 62.5%, 1 s recovery 93.5%, against 60.5% and 87.1% undamped) and is about even at 1280×736, but it adds ID switches at 640: 573 at damping 0.8 and 580 at 0, against 557 committed and 531 undamped. So it stays off.

Most of the 640 cost is on two moving-camera videos. At damping 0.8 + cap 0.75, MOT17-05 goes from 81 to 96 ID switches and MOT17-10 from 160 to 173, against the committed run. At 1280×736 on MOT17-05, damping 0.8 alone (no gate) already adds 12 of the 20 extra ID switches there (75 → 87). Our guess, not checked: with the camera moving, a person's image motion is mostly the camera's and stays steady, so slowing the prediction down puts it behind them.

Choosing between settings partly by MOT17 overfits to MOT17 a little. That's another reason to take 0.5 over 0.75: it wasn't tuned.

## How often the cap binds

Share of gate checks (a searching track reaching the gate stage with leftover detections) where the cap is below the full gate, and where a detection lay between the two, so the cap changed what could match. Scratch wrapper around the tracker, not committed.

| Setting | Anti-UAV binds | Anti-UAV changes the outcome | MOT17 1280×736 binds | MOT17 640 binds |
|---|---|---|---|---|
| cap 0.5, undamped (default) | 29.5% | 9.1% | 95.7% | 95.7% |
| cap 0.75, undamped | 23.5% | 8.0% | 94.2% | 94.1% |
| cap 0.5, damping 0.8 | 50.9% | 22.5% | 99.4% | 99.6% |
| cap 1.0, damping 0.8 | 41.3% | 18.5% | 97.5% | |

On MOT17 the cap binds almost every time, so it is what keeps MOT17 flat: uncapped, the same gate gives 1261 ID switches at 1280×736.

**On Anti-UAV it binds far more often than the issue expected ("it should almost never").** The drone isn't alone in the sky: the tracker's own duplicate IDs of it are. Each time the cap bound (damping 0.8, cap 0.5, 20% deletion), the nearest other confirmed track was:

| Nearest track | Binds | Changed the outcome |
|---|---|---|
| searching, younger than this track | 1,485 | 859 |
| searching, older | 1,188 | 292 |
| tracked, younger | 309 | 181 |
| tracked, older | 60 | 22 |

Median distance to it: 2.3 box sizes (10th / 90th percentile 0.44 / 7.5). A younger searching track is a duplicate the tracker made of the same drone, most likely from a tentative track that got confirmed while the original was searching. Two searching tracks on one drone shrink each other's gate, which is likely most of the 1–2 points of ID kept the cap costs. Fixing the duplicates (say, merging a searching track into a younger one that sits on its prediction) would help both, but isn't part of this issue.

## Reproduce

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val/cap/default --name default
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name default_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n.onnx --name default_yolo26n
tether-antiuav data/Anti-UAV300 --split val --drop-rates 0 0.2 --gap-seconds 1.0 --work-dir results/antiuav-val/cap/d0.8 \
    --name d0.8 --tracker "tether-track --coast-damping 0.8 --search-gate-cap 0.75"
```

The old behavior is `tether-track --search-gate 0 --search-gate-growth 0`. Grid rows used `--drop-rates 0 0.2 --gap-seconds 1.0` through a scratch wrapper that also counted the binds (it only counts; the tracking is the committed code).
