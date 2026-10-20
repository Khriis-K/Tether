# Anti-UAV reachable ceiling

Run 2026-10-20 for #33. The best "ID kept" any tracker could score on Anti-UAV300, by how long a track searches before it's forgotten. No tracker is run: this is an upper bound computed from the labels alone. Val is 67 clips, test 91, all at 20 fps, so 2 s is 40 frames and 3 s is 60. Deletion is seed 0, the same boxes `tether-antiuav` deleted (the scored frame counts match the tracker runs exactly).

## Results

ID kept ceiling, in % of scored frames:

| Search window | val, 0% deleted | val, 20% deleted | test, 0% deleted | test, 20% deleted |
|---|---|---|---|---|
| 30 frames (1.5 s, today's default) | 96.5 | 96.5 | **94.2** | **94.0** |
| 2 s (40 frames) | 97.1 | 97.1 | 94.7 | 94.6 |
| 3 s (60 frames) | 97.8 | 97.8 | 97.2 | 97.2 |
| No limit | 100.0 | 100.0 | 100.0 | 100.0 |
| Scored frames | 58,302 | 46,788 | 79,724 | 63,723 |

- **The 95% target is below the ceiling on val but above it on test**, at the 30-frame default and at 2 s. Only from 3 s does test clear it, by 2.2 points.
- **Deletion barely moves the ceiling** (at most 0.2 points). Random 20% deletion almost never makes a run of 31+ missing frames on its own (0.2³¹); it only lengthens absences that were already near the limit. What limits the ceiling is how long the drone leaves the labels, not the deletions.
- **Time-based vs. frames makes no difference here**: every clip is 20 fps. It would only matter on other footage.
- Val and test disagree by up to 2.3 points at the same window, so a target set from val alone could still miss on test.

## Caveats

- **It's an upper bound, not a prediction.** It assumes a tracker that never loses the ID except when forced: every scored frame inside the clip's longest stretch counts, ignoring the 3 frames a new track needs to be confirmed and the IoU ≥ 0.5 rule. Tether scores 85.3% at 20% deletion on val against 96.5% here (`antiuav-duplicates.md`).
- **"No limit" at 100% is trivial by construction.** On single-drone clips, a track that never dies can't lose the ID to an absence. The cost of a long window is elsewhere and not measured here: in crowds (MOT17), a long search gives a dead track more time to be revived on the wrong person.
- One deletion seed. Since deletion barely matters, more seeds are unlikely to move these numbers, but that's not measured.

## How it's scored

`id_kept_ceiling` in `python/tether/antiuav.py`: per clip, split the given boxes wherever more than *window* frames in a row have no box (unlabeled or deleted); the kept frames are the given boxes in the longest piece, over all given boxes. Summed over clips. Same scored frames as `id_kept`: only frames whose box was given to the tracker.

## Reproduce

```
python tools/antiuav_ceiling.py data/Anti-UAV300 --split val --work-dir results/antiuav-val
python tools/antiuav_ceiling.py data/Anti-UAV300 --split test
```

Results go to `<work-dir>/ceiling/<split>.json`. Each run takes a few seconds. Test at 0% deletion and 30 frames reproduces the 94.2% in `antiuav-dropout.md`.
