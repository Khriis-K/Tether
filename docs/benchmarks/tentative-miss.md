# Letting tentative tracks survive a miss

Run 2026-10-19 for #31. Same setup as [antiuav-duplicates.md](antiuav-duplicates.md): Anti-UAV **val** (67 clips, 20 fps), seed 0. MOT17 is the 7 train videos with the dumped YOLO26n detections, at 1280Ã—736 and 640. #29 (the growing gate) and #36 (the duplicate merge) are both on `main`, so "before" is `main` after #36.

**Result: shelved, not merged.** A tentative track that survives one miss and confirms on net hits cuts the IDs handed out on Anti-UAV by a fifth, but the score doesn't move: ID kept at 20% deletion goes from 85.3% to 85.2%. MOT17 at 640 regresses (HOTA 33.13 â†’ 32.67, IDF1 37.62 â†’ 37.02, ID switches 529 â†’ 536), and 1280Ã—736 is flat in total with MOT17-09 down 0.50 HOTA. The code and its tests are kept on the branch `issue-31-tentative-miss-experiment`.

## The change

A tentative track survived no misses: it needed `confirm_frames` (3) hits in a row, so one deleted box in its first frames dropped it, and the next detection started another ID. The [leaky counter](../CONCEPTS.md#leaky-counter-tried-shelved) tried here (`--tentative-misses`, default 1 on the branch):

- A tentative track survives one missed frame in a row and is dropped on the second.
- It is confirmed when its **net hits**, frames matched minus frames missed, reach `confirm_frames`. A one-off false detection never gets past 1, and one that flickers every other frame stays at 0â€“1.
- Two follow-ons in the duplicate merge. A merged tentative is dropped, where it would otherwise survive its "miss" as a ghost on top of the track that took it over. And a track only takes over a *younger* one. A tentative that survives misses can now outlive a track born and confirmed after it, which is the age caveat in antiuav-duplicates.md.

## Measure first

At `main`, tracking in-process (scratch, `results/i31/measure.py`, `fate.py`, `why.py`, not committed). Every new ID after a clip's first, by what was alive when it was born:

| | 0% deleted | 20% deleted |
|---|---|---|
| New IDs | 1668 | 1925 |
| ...born while a confirmed track was alive (the #36 case) | 1639 (98.3%) | 1863 (96.8%) |
| ...born after a tentative track died, nothing alive | 14 (0.8%) | 47 (2.4%) |
| ...born after a confirmed track expired, nothing alive | 15 (0.9%) | 15 (0.8%) |
| ...of those 15, the drone unlabeled for over 30 frames first | 13 | 13 |

Nearly every birth happens while the drone's confirmed track is still searching. #36 merges those back into the old ID once they confirm, so they mostly don't cost an ID; they cost the frames until the merge. What happens to the tentative tracks themselves:

| Tentative track | 0% deleted | 20% deleted |
|---|---|---|
| Merged into the old track (#36) | 868 | 717 |
| Died: its detection was deleted | 3 | 637 |
| Died: an older track took the detection back (the gate) | 598 | 389 |
| Died: the detection was there but went unmatched | 167 | 145 |
| Confirmed under its own ID | 96 | 103 |

**637 tentative tracks at 20% deletion died only because their box was deleted, against 15 confirmed tracks expiring.** By the issue's test ("if tentative deaths are a small share, close"), that isn't small, so it was built.

The issue asked for this before and after #29. #29 had already landed, so only "after" was measured.

## Results

Committed code on the branch, plain `tether-track` (from a frozen copy of the code, see Caveats):

| | ID kept 0% | ID kept 20% | 0.25 s gaps | 0.5 s gaps | 1 s gaps |
|---|---|---|---|---|---|
| `main` (#36) | 88.4% | 85.3% | 92.0% | 88.0% | 88.8% |
| tentative survives a miss | 88.4% | 85.2% | 92.0% | 87.9% | 88.8% |

From the in-process scripts, at 0% / 20% deletion:

| | `main` | tentative survives a miss |
|---|---|---|
| New IDs after each clip's first | 1668 / 1925 | 1566 / 1508 |
| ...later confirmed under their own ID | 32 / 59 | 32 / 42 |
| Scored frames with two or more confirmed tracks | 0.5% / 0.8% | 0.5% / 0.7% |
| Confirmed false tracks (never hold the drone) | 0 / 0 | 0 / 0 |

**Confirmed false tracks on Anti-UAV can't tell us anything:** every detection is the labeled drone, so a confirmed track almost has to hold it at some point. That's the null case, and it reads 0 either way. MOT17 is the real check, as the issue said. There, counting output IDs that never overlap a ground-truth person (IoU â‰¥ 0.5, scratch `results/i31/mot_false.py`):

| | 1280Ã—736 | 640 |
|---|---|---|
| `main` (#36) | 211 of 1003 IDs (30.1 per video) | 93 of 682 (13.3 per video) |
| tentative survives a miss | 212 of 1005 (30.3 per video) | 95 of 685 (13.6 per video) |

So the leaky counter doesn't confirm noticeably more false tracks.

**MOT17**, HOTA / IDF1 / ID switches:

| | 1280Ã—736 | 640 |
|---|---|---|
| `main` (#36) | 41.46 / 49.38 / 732 | 33.13 / 37.62 / 529 |
| tentative survives a miss | 41.47 / 49.42 / 733 | 32.67 / 37.02 / 536 |

Per video, `main` â†’ tentative survives a miss:

| Video | Size | HOTA | IDF1 | ID switches |
|---|---|---|---|---|
| MOT17-02 | 1280Ã—736 | 27.49 â†’ 27.62 | 32.05 â†’ 32.43 | 136 â†’ 139 |
| MOT17-04 | 1280Ã—736 | 45.62 â†’ 45.74 | 54.16 â†’ 54.39 | 177 â†’ 174 |
| MOT17-05 | 1280Ã—736 | 42.66 â†’ 42.52 | 58.39 â†’ 58.21 | 76 â†’ 74 |
| MOT17-09 | 1280Ã—736 | 44.59 â†’ 44.09 | 52.69 â†’ 51.50 | 67 â†’ 67 |
| MOT17-10 | 1280Ã—736 | 35.85 â†’ 35.73 | 44.87 â†’ 44.63 | 134 â†’ 130 |
| MOT17-11 | 1280Ã—736 | 49.09 â†’ 49.09 | 52.47 â†’ 52.44 | 36 â†’ 36 |
| MOT17-13 | 1280Ã—736 | 37.51 â†’ 37.47 | 49.00 â†’ 48.95 | 106 â†’ 113 |
| MOT17-02 | 640 | 23.92 â†’ 23.89 | 25.88 â†’ 25.85 | 57 â†’ 58 |
| MOT17-04 | 640 | 31.51 â†’ 31.60 | 33.97 â†’ 34.24 | 64 â†’ 63 |
| MOT17-05 | 640 | 42.33 â†’ 42.31 | 55.81 â†’ 55.76 | 76 â†’ 76 |
| MOT17-09 | 640 | 39.01 â†’ 38.07 | 45.11 â†’ 43.74 | 64 â†’ 68 |
| MOT17-10 | 640 | 30.84 â†’ 29.30 | 38.95 â†’ 36.18 | 152 â†’ 151 |
| MOT17-11 | 640 | 48.11 â†’ 45.98 | 52.84 â†’ 49.97 | 35 â†’ 37 |
| MOT17-13 | 640 | 27.82 â†’ 27.85 | 34.51 â†’ 34.62 | 81 â†’ 83 |

## Why ID kept didn't move

Two reasons, the first from the arithmetic and the second from the measurements above.

- **Charging a full hit per miss makes a survived miss worth at most one frame.** Hit, miss, hit, hit, hit confirms on the fifth frame, the same frame as a fresh track started on the third. Only a miss after two hits saves anything: hit, hit, miss, hit, hit confirms on the fifth frame, one sooner than a fresh track started on the fourth. So surviving doesn't confirm the drone much sooner; it mostly keeps the same number.
- **#36 already keeps the ID.** A tentative track born while the drone's track searches is merged back into the old ID when it confirms. Keeping that tentative's own number alive is worth nothing to ID kept, which scores the old ID.

So the 2+ confirmed-track frames the issue comment worried about barely changed (0.8% â†’ 0.7% at 20%), and the IDs saved were IDs nobody scored.

## Why MOT17 at 640 regressed

**Not found.** The obvious suspect was flickery false positives getting confirmed, and false tracks barely changed (93 â†’ 95), so it isn't that. Our guess, not checked: a one-hit tentative that survives a miss still gets the center-distance second chance a frame later (stage 5), from a box that hasn't moved, so in a crowd it can take a nearby person's new detection. MOT17-10 and MOT17-11 at 640 lose the most (âˆ’1.54 and âˆ’2.13 HOTA). Since Anti-UAV shows nothing to win, the cause wasn't chased.

## Caveats

- The leaky counter here counts misses in a row for survival and net hits for confirmation. Norfair's own counter drops a track when net hits fall to zero. With that rule, hit, miss would drop a one-hit track, which is the old behavior in the case #31 was about, so the issue's "survives one miss" was followed instead. A flicker every other frame then lives on as a tentative track forever. Evaluation ignores tentative tracks, so no number here counts it.
- Only one survival setting (one miss) was measured. Confirming on total hits instead of net hits would confirm sooner, but it would also confirm a detection that flickers every other frame, which the issue ruled out.
- The first benchmark runs overlapped the mutation testing in `docs/spec-verify/issue-31.md`, which edits `tracker.py` in place, and both harnesses start a fresh tracker process per clip. Everything above was rerun from a frozen copy of the code (`results/i31/snapshot/`). The MOT17 reruns matched the first runs exactly.
- 1 s gaps are recovery over 874 held gaps of 928, the same count as on `main`.
- Val only; test stays unseen.

## Reproduce

On the branch `issue-31-tentative-miss-experiment`:

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/antiuav-val/i31 --name default
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name i31_default_yolo26n_1280x736
tether-mot17 data/MOT17 --model models/yolo26n.onnx --name i31_default_yolo26n
```

The `main` behavior on the branch is `tether-track --tentative-misses 0`. Birth causes, tentative fates, false tracks: `results/i31/measure.py`, `fate.py`, `why.py`, `mot_false.py` (scratch, not committed).
