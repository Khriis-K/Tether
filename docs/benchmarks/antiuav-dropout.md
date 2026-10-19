# Anti-UAV dropout tests

Run 2026-10-18 for #8. Tether (Python, default settings: confirm after 3 frames, search for 30) on the 91 visible-light clips of the Anti-UAV300 test split, 1920×1080 at 20 fps, 85,374 frames. No detector: the labeled drone boxes are the detections (score 1.0), with failures added on purpose by `tether-antiuav` (`python/tether/antiuav.py`). Seed 0.

## Results

| Test | Result | SPEC target |
|---|---|---|
| Nothing deleted (baseline) | ID kept on **37.6%** of 79,724 frames | — |
| 20% of boxes deleted at random | ID kept on **26.1%** of 63,723 frames | at least 95% |
| Gaps of 0.25 s (5 frames) | **68.5%** recovered (1,100 of 1,607 held; 1,741 gaps) | — |
| Gaps of 0.5 s (10 frames) | **48.3%** recovered (676 of 1,399 held; 1,511 gaps) | — |
| Gaps of 1 s (20 frames) | **27.2%** recovered (325 of 1,196 held; 1,268 gaps) | at least 80% |

Prediction error during gaps (not pass/fail):

| Gap | Gap frames measured | Mean center error | Mean IoU with the drone |
|---|---|---|---|
| 0.25 s | 8,028 | 33.2 px | 0.71 |
| 0.5 s | 13,980 | 73.3 px | 0.57 |
| 1 s | 23,832 | 141.8 px | 0.42 |

Both targets are missed by a wide margin. Most of the loss is already there with nothing deleted.

**What's reachable.** The drone sometimes leaves the labels for longer than a track searches (30 frames, 1.5 s here), and after that a new ID is unavoidable for Tether. Counting each clip's longest stretch without such an absence, the best possible "ID kept" with nothing deleted is **94.2%**. The 95% target sits just above that for a tracker that forgets after 30 frames. It is not the gap that matters, though: Tether scores 37.6% against 94.2%.

## How it's scored

- **Holding the drone.** A track holds the drone on a frame only if it is `"tracked"` and its box has IoU ≥ 0.5 with the labeled box. A searching track's box is where it was last seen, so it never counts. Frames where the drone isn't labeled (`exist` 0) are never scored.
- **ID kept.** Only frames whose box was given to the tracker are scored (otherwise 20% deletion would cap the score at 80%). Score: the share of those frames held by the clip's most common ID. Counts are summed over clips.
- **Gaps.** Each gap follows 2 s of detections, then repeats: settle, gap, settle, gap. Gap length in seconds is converted with each clip's frame rate. A gap is scored if the drone is labeled on the frame before it and on some frame before the next gap. "Held" gaps are the ones a track held the drone on the frame before; recovery is the share of held gaps where the same ID holds it on the first labeled frame after. Unheld gaps are counted but not scored, so a tracker that outputs nothing gets no rate rather than a flattering one.
- **Prediction error.** On each frame of a held gap where the drone is labeled: the held track's `predicted_box` from the frame before, against the labeled box. Only tracks still alive are measured; with gaps of 1 s or less, no track is removed mid-gap at the default search limit.

## Why Tether loses the drone

The numbers in this section come from `tools/antiuav_diagnose.py` (see Reproduce).

**It finds the drone but loses the identity.** With nothing deleted, some track holds the drone on 93.0% of frames, but the most common ID per clip only on 37.6%. One 1,000-frame clip (`20190925_124000_1_3`) used 81 IDs.

**A single missed prediction is enough.** Frame to frame, Tether predicts well: the track holding the drone predicts the next box below IoU 0.3 (the matcher's minimum) on only 3.4% of 73,971 frames. That's about as good as a perfect constant-velocity guess from the labels (`2·box[f−1] − box[f−2]`, 3.9% of 79,436 frames), and better than "stay where it was" (6.3%). But each of those misses costs the ID. The track goes searching and coasts on its old velocity, so its prediction moves further from the drone each frame and IoU matching never picks it up again. Meanwhile the drone's detection starts a new track. At 3.4% a frame, that's an ID break about every 30 frames, which is what the baseline shows.

**Why the predictions miss: the camera hunts.** The labeled boxes are right: the drone sits inside them on every frame checked. But its image motion is a sawtooth. The camera is a pan-tilt unit keeping the drone near a burned-in crosshair, so the drone drifts about 28 px a frame one way for 8–10 frames, then snaps back at about 50 px a frame as the camera re-centers. In `20190925_134301_1_2`, the labeled box's left edge goes 721 → 987 px over frames 0–11, then back to 724 by frame 17. Each snap-back is a frame a constant-velocity prediction can miss.

**Deletions and gaps make it worse.** A deleted frame means coasting through the miss, and the gap errors above grow about linearly with gap length (33 → 73 → 142 px), as coasting on a fixed velocity would. Deletions also hit new tracks: a tentative track needs 3 hits in a row, so a deletion in either of its next 2 frames (a 36% chance at 20%) drops it, and the next detection starts yet another. With 20% deleted, some track holds the drone on 87.8% of scored frames, the most common ID on 26.1%.

So the first thing to fix is re-matching a track that has just missed, not prediction itself: the detection is still close, but IoU against a coasted box can't see it. `docs/research/camera-motion.md` collects options (re-association by a distance gate that grows while searching, velocity damping while coasting, tentative tracks that survive a miss, offline camera-motion compensation). Not measured yet: how ByteTrack does on the same files, and how much each change would help. Any tuning goes on the train or val split; test stays for final numbers.

## Reproduce

```
pip install -e .[demo]
tether-antiuav data/Anti-UAV300
python tools/antiuav_diagnose.py data/Anti-UAV300 results/antiuav/detections/drop-0.0-seed0 results/antiuav/trackers/tether/drop-0.0-seed0
python tools/antiuav_diagnose.py data/Anti-UAV300 results/antiuav/detections/drop-0.2-seed0 results/antiuav/trackers/tether/drop-0.2-seed0
```

`data/Anti-UAV300` is `Anti-UAV-RGBT.zip` from the [Anti-UAV repository](https://github.com/ZhaoJ9014/Anti-UAV) (5.6 GB; only the `visible.*` files and `label_new/` are needed). Defaults: `--split test --drop-rates 0 0.2 --gap-seconds 0.25 0.5 1.0 --seed 0`. Any tracker command can be scored with `--tracker` and `--name`, as for `tether-mot17`. Detection files are saved under `results/antiuav/detections/<condition>/`, so other trackers run on the very same input. Results go to `results/antiuav/trackers/<name>/results.json`, diagnostics to `results/antiuav/diagnose/<condition>.json`. A run takes about 24 minutes, nearly all of it the tracker. A rerun with the same seed reproduced every number.
