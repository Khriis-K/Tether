# Single-target prior: an upper bound on re-association

Run 2026-10-20 for #34. Same setup as [antiuav-ceiling.md](antiuav-ceiling.md): Anti-UAV **val** (67 clips, 20 fps), seed 0, the default 30-frame search. MOT17 is the 7 train videos with the dumped YOLO26n detections, at 1280×736 and 640. "Default" is `main` after #33, re-run in the same work dir; it reproduces the committed 85.3%.

**These are upper bounds, not results.** The option knows there's one drone per clip, which is false for general tracking. Its numbers never count toward #28's targets, and it never becomes a default or goes into the C++ port.

**Result: a perfect re-association of searching tracks recovers 7.9 of the 11.2 points (71%) between Tether and the ceiling at 20% deletion (85.3% → 93.3%, ceiling 96.5%).** Of the 3.2 points left, 2.8 are frames where the drone jumps away from a track that was still "tracked" (the rule only helps from the next frame), and 0.3 are each clip's first track being confirmed. On MOT17 the rule never fires, so it changes nothing.

## The option

`Tracker(single_target=True)`, or `tether-track --single-target`, off by default. At the start of a frame, if exactly one confirmed track exists and it's searching, then after stage 3 (the search gate) it takes the nearest high detection left anywhere in the frame. There's no IoU, gate, cap or size limit, and it runs before tentative tracks can take that detection. It never fires while any track is tracked, or while two or more confirmed tracks exist. Tentative tracks don't block it, though the issue's wording ("more than one track exists") would have them do so. A missed jump starts a tentative track on the drone, so blocking on tentative tracks would block the very case the oracle is for. "Nearest" is center distance in pixels: for one track that ranks detections the same as box sizes do, and on val a long coast turned one predicted box inside out, which left it no size to divide by.

With defaults, 15 existing detection files (3 MOT17 videos, 8 Anti-UAV clips at 20% deletion and 4 with 1 s gaps) give byte-identical track files on `main` and with the option added.

## Results

Anti-UAV val, ID kept in % of scored frames:

| | 0% deleted | 20% deleted |
|---|---|---|
| Default (counts toward #28) | 88.4 | **85.3** |
| Single target (upper bound, does not count) | 93.1 | **93.3** |
| Reachable ceiling, 30-frame window (`antiuav-ceiling.md`) | 96.5 | **96.5** |
| Points recovered, of the gap to the ceiling | 4.68 of 8.12 (57.7%) | 7.95 of 11.17 (71.1%) |
| Share of the ceiling: default → single target | 91.5% → 96.4% | 88.4% → 96.6% |

Gap recovery, upper bound: 0.25 s gaps 92.0% → 100.0%, 0.5 s 88.0% → 99.9%, 1 s 88.8% → 99.8%.

**Against the target:** #28's ID-kept target is 95% of the ceiling, 91.7% on val. Default needs 2,971 more kept frames to reach it, and the oracle gains 3,718. So a general re-association rule, one that doesn't know there's one drone, would have to recover 79.9% of what this oracle recovers to meet the target by itself.

## Where the gap goes

Every frame the ceiling keeps but the clip's main ID (its most common holder) doesn't hold, put in one bucket using that frame and the one before. In points of scored frames, 20% deleted (0% deleted in brackets):

| Cause | Default | Single target |
|---|---|---|
| Another ID holds the drone | 4.66 (3.50) | 0.00 (0.00) |
| Main ID searching last frame, still not back (re-association miss) | 3.53 (1.82) | 0.04 (0.03) |
| Main ID tracked last frame, loses the drone this frame (jump) | 1.74 (2.23) | 2.80 (3.16) |
| The main ID's own first frames, before it's confirmed | 0.29 (0.23) | 0.29 (0.23) |
| Other (main ID not alive, nobody else holds it) | 0.96 (0.34) | 0.10 (0.01) |
| **Total = ceiling − ID kept** | **11.17 (8.12)** | **3.23 (3.44)** |

- **Re-association is 8.19 of the default's 11.17 points** (another ID, plus re-association misses), and the oracle takes it to zero. It recovers 7.95 net, because some of those frames turn into jump frames instead (below).
- **What's left is mostly jumps.** On 1,308 frames (2.80 points) the drone moved so far between frames that the track, still "tracked", fails the IoU match. The rule waits for the track to become searching, so it takes the drone back on the next frame, and each jump costs one frame. The default has fewer frames in this bucket (816), because there a jump often goes on to become a re-association miss or a new ID, which land in the other buckets.
- **Confirmation costs 134 frames**: 2 per clip, while the clip's first track waits for its 3rd hit. No tracker with `confirm_frames` 3 avoids them, and the ceiling ignores them (its own caveat).
- **"Frames where no track holds the drone"** are not a cause by themselves: with the oracle, frames where neither a tracked nor a tentative track overlaps the drone are 0.03 points.

## MOT17

HOTA / IDF1 / ID switches, default vs. `--single-target`, all 7 videos:

| | 1280×736 | 640 |
|---|---|---|
| Default | 41.46 / 49.38 / 732 | 33.13 / 37.62 / 529 |
| Single target | 41.46 / 49.38 / 732 | 33.13 / 37.62 / 529 |

The two runs' track files are byte-identical. The rule's condition (exactly one confirmed track, and it's searching) holds on **0 of the 5,316 frames** at either size, because a crowd always has more than one confirmed track. So it's meaningless on MOT17 rather than harmful. On footage with one person in view it would fire, and there it would hand a lost person's ID to anyone who walks in: the sparse-scene swap from #37 with no limit on distance at all.

## Caveats

- **Upper bound, val only.** Tuning happens on val, and test stays reserved for #28's one final run, so the oracle was not run on test.
- **The oracle is "perfect" only for searching tracks.** It doesn't help a track that is still tracked when the drone jumps (the 2.8 points above). A rule that let a tracked track that fails IoU take the detection at once might close part of that, but it wasn't run, so we don't know.
- One deletion seed. The default reproduces the committed numbers exactly, so the comparison is like for like.
- "Nearest" is the only notion of "best" tried. On Anti-UAV each frame has at most one detection, so it doesn't matter there.

## Reproduce

```
tether-antiuav data/Anti-UAV300 --split val --work-dir results/i34 --name default
tether-antiuav data/Anti-UAV300 --split val --work-dir results/i34 --name single_target --tracker "python -m tether.cli --single-target"
tether-mot17 data/MOT17 --model models/yolo26n_1280x736.onnx --name i34_single_target_yolo26n_1280x736 --tracker "python -m tether.cli --single-target"
tether-mot17 data/MOT17 --model models/yolo26n.onnx --name i34_single_target_yolo26n --tracker "python -m tether.cli --single-target"
```

Gap causes, firing counts on MOT17: `breakdown.py` and `causes.py` (scratch, not committed).
