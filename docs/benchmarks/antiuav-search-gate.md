# Growing search gate for searching tracks

Run 2026-10-19 for #29. Same setup as [antiuav-dropout.md](antiuav-dropout.md), but on the **val** split (67 clips, 20 fps; test stays reserved), seed 0. MOT17 is the 7 train videos with the dumped YOLO26n detections, as in [mot17-bytetrack.md](mot17-bytetrack.md).

The change: a matching stage where searching tracks still unmatched after IoU are re-matched to leftover high detections by center distance, within `search_gate + search_gate_growth · frames_since_seen` track box sizes. It runs before tentative tracks, so the tentative track that a missed detection started can't keep it. It reuses `match_by_distance`, which now takes one gate per track.

**Result: it is in the code but off by default.** Every setting that helps Anti-UAV makes MOT17 worse, so nothing changes until a gate is passed (`tether-track --search-gate R0 --search-gate-growth K`). With the defaults, track files are byte-identical to main's on all 81 files checked (every MOT17 detection file at both input sizes, plus the val 20%-deletion files).

## Val baseline and ByteTrack

| Tracker | ID kept, 0% deleted | ID kept, 20% deleted | 0.25 s gaps | 0.5 s gaps | 1 s gaps |
|---|---|---|---|---|---|
| Tether (main, no gate) | 48.3% | 30.5% | 72.7% | 54.4% | 31.3% |
| ByteTrack | 49.2% | 33.6% | 73.3% | 56.8% | 38.0% |
| Tether, gate 1.0 + 0.25/frame | 57.2% | 57.9% | 90.2% | 86.1% | 86.8% |
| Tether, gate 1.5 + 0.25/frame | **60.8%** | **61.1%** | **92.0%** | **87.6%** | **88.4%** |
| SPEC target | — | at least 95% | — | — | at least 80% |

Gap columns are recovery rates (the share of held gaps where the same ID holds the drone after). ID kept is scored over 58,302 frames (0%) and 46,788 (20%); 1 s gaps over 866–885 held gaps of 928.

- **ByteTrack loses the drone just like Tether does.** It is 0.6–7 points above Tether's baseline on every test, so the failure isn't specific to Tether. (ByteTrack's prediction error isn't comparable: our adapter outputs only tracked tracks, so only the first frame of each gap is measured.)
- **The gate meets the 1 s recovery target on val (88.4% vs 80%), but not ID kept (61.1% vs 95%).** The best ID kept is still a long way below the 94.2% reachable on test (not computed for val).

## Tuning on val

Val with 0% and 20% deleted, plus 1 s gaps; MOT17 at 1280×736. The gate is in units of the track box's larger side.

| Gate r0 | Growth k | ID kept 0% | ID kept 20% | 1 s gaps | MOT17 HOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|---|
| off (main) | | 48.3% | 30.5% | 31.3% | 41.29 | 49.04 | 748 |
| 0.5 | 0 | 53.1% | 36.8% | 37.5% | 41.04 | 49.24 | 765 |
| 0.5 | 0.05 | 53.8% | 39.8% | 55.7% | 40.96 | 48.43 | 882 |
| 0.5 | 0.1 | 54.5% | 41.5% | 65.0% | | | |
| 0.5 | 0.25 | 55.5% | 51.8% | 85.0% | 39.53 | 46.14 | 1102 |
| 1.0 | 0 | 55.8% | 52.8% | 49.1% | | | |
| 1.0 | 0.05 | 56.1% | 55.1% | 60.8% | 40.17 | 47.42 | 965 |
| 1.0 | 0.1 | 56.5% | 56.1% | 69.6% | | | |
| 1.0 | 0.25 | 57.2% | 57.9% | 86.8% | 39.43 | 45.85 | 1164 |
| 1.0 | 0.5 | | | | 38.15 | 44.05 | 1328 |
| 1.5 | 0 | 57.9% | 56.9% | 55.4% | | | |
| 1.5 | 0.05 | 58.4% | 58.1% | 64.8% | | | |
| 1.5 | 0.1 | 58.3% | 58.3% | 73.1% | | | |
| 1.5 | 0.25 | 60.8% | 61.1% | 88.4% | 38.88 | 44.92 | 1261 |

At 640, the two settings checked also regress: gate 0.5 + 0 gives 32.37 HOTA / 36.57 IDF1 / 598 ID switches, and 1.0 + 0.25 gives 30.33 / 33.63 / 840, against main's 32.82 / 36.95 / 557.

- **On Anti-UAV, wider is better almost everywhere in the grid** (one exception: 1.5 + 0.1 gives 58.3% ID kept at 0% deleted, against 58.4% for 1.5 + 0.05). The starting gate r0 mostly drives ID kept, and the growth k drives gap recovery.
- **On MOT17, every gate costs something.** Even the smallest (0.5, no growth) loses 0.25 HOTA and adds 17 ID switches at 1280×736, and 0.45 HOTA and 41 ID switches at 640. With growth the cost climbs fast: 1.0 + 0.25 adds 416 ID switches at 1280×736.

### Variants that didn't fix the MOT17 cost

These went through a throwaway copy of the stage (not committed), MOT17 at 1280×736:

| Variant | Settings | HOTA | IDF1 | ID switches |
|---|---|---|---|---|
| Tracks tracked last frame get the gate too (at r0) | 1.0 + 0.5 | 37.29 | 42.47 | 1134 |
| That, and the stage after tentative tracks | 0.5 + 0.05 | 40.70 | 47.80 | 853 |
| Distance per axis (dx in widths, dy in heights) | 0.5 + 0.05 | 40.02 | 47.54 | 770 |
| Gate × the track's own speed (box sizes/frame) | 2 + 0.5 | 41.37 | 49.18 | 786 |

The speed-scaled gate is the only one near main on MOT17, but on val it keeps little of the gain. Settings 2 + 0.5 give 50.1% / 32.2% ID kept and 38.4% of 1 s gaps; the best tried (4 + 1) gives 52.5% / 39.4% / 59.9%.

## Why it doesn't transfer

- **On Anti-UAV a wrong match is impossible.** The only detection on a frame is the labeled drone, so a "tracked" box always overlaps it fully. That makes the issue's false-match check (tracked boxes with IoU < 0.5) zero by construction, before and after. It can't tell a good gate from a reckless one. MOT17's ID switches are where wrong matches show, and they rise with every gate.
- **The unit is wrong for people.** The gate is in box sizes, and per frame a drone here moves about 0.3–0.6 of its box (28–50 px against a median box of about 86 px, from antiuav-dropout.md and camera-motion.md). A walking pedestrian moves a much smaller share of their height (not measured here), and the larger side of a person's box is their height: if it's about 2.5 times the width (typical, not measured here), a gate of r0 reaches 2.5·r0 widths sideways, where the neighbors are. A gate big enough for the drone's sawtooth reaches a neighbor in a crowd. Speed-scaling fixes the unit for MOT17 but is too tight for the sawtooth, where the miss is drift plus snap-back, not the track's own speed.

## What still breaks the ID on Anti-UAV

Each time the drone's holder changes, here is what the old holder was doing (val, nothing deleted; scratch script, not committed):

| | Holder IDs | Old holder searching, 2–5 frames missed | 1 frame missed | > 5 missed | Removed |
|---|---|---|---|---|---|
| main | 1,418 | 1,334 | 15 | 18 | 15 |
| gate 1.5 + 0.25 | 797 | 739 | 328 | 9 | 15 |

The gate almost halves the IDs, but most handovers still happen 2–5 frames into a search. That's when the tentative track from the missed detection would get confirmed (3 hits; the script doesn't check the new holder's hits). So the searching track likely couldn't catch the drone even inside a gate of 1.75–2.5 box sizes (the counts are the frames missed as reported after the frame, one more than when the gate was applied). The coasted prediction moves away from the drone after a sawtooth reversal, at about the reversal speed. That's what velocity damping (shortlist #2 in [camera-motion.md](../research/camera-motion.md)) would address, and it would make a smaller gate (one MOT17 might tolerate) enough. The single-frame miss distance, from the old holder's prediction to the drone, is 0.5 / 0.95 / 2.8 / 6.4 box sizes at the 10th / 50th / 90th / 99th percentile.

The new "1 frame missed" handovers (328) are another *confirmed* track taking the drone, most likely an older duplicate ID that is also searching and happens to be closer.

## Reproduce

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val --name tether
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val --name bytetrack --tracker tether-bytetrack
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val --name tether-gate-1.5-0.25 \
    --tracker "tether-track --search-gate 1.5 --search-gate-growth 0.25"
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name gate-1.0-0.25_yolo26n_1280x736 \
    --tracker "tether-track --search-gate 1.0 --search-gate-growth 0.25"
```

Grid rows used `--drop-rates 0 0.2 --gap-seconds 1.0`. A full val run takes about 30 minutes for Tether and longer for ByteTrack. The grid numbers match the full runs exactly where they overlap.
