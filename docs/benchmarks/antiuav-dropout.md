# Anti-UAV dropout tests

Run 2026-10-18 for #8. Tether (Python, default settings: confirm after 3 frames, search for 30) on the 91 visible-light clips of the Anti-UAV300 test split, 1920×1080 at 20 fps, 85,374 frames. No detector: the labeled drone boxes are the detections (score 1.0), with failures added on purpose by `tether-antiuav` (`python/tether/antiuav.py`). Seed 0.

## Results

| Test | Result | SPEC target |
|---|---|---|
| 20% of boxes deleted at random | ID kept on **26.1%** of 63,723 frames | at least 95% |
| Gaps of 0.25 s (5 frames) | **68.5%** recovered (1,100 of 1,607) | — |
| Gaps of 0.5 s (10 frames) | **48.3%** recovered (676 of 1,399) | — |
| Gaps of 1 s (20 frames) | **27.2%** recovered (325 of 1,196) | at least 80% |

Prediction error during gaps (not pass/fail):

| Gap | Gap frames measured | Mean center error | Mean IoU with the drone |
|---|---|---|---|
| 0.25 s | 8,028 | 33.2 px | 0.71 |
| 0.5 s | 13,980 | 73.3 px | 0.57 |
| 1 s | 23,832 | 141.8 px | 0.42 |

Both targets are missed by a wide margin.

## How it's scored

- **Holding the drone.** A track holds the drone on a frame only if it is `"tracked"` and its box has IoU ≥ 0.5 with the labeled box. A searching track's box is where it was last seen, so it never counts. Frames where the drone isn't labeled (`exist` 0) are never scored.
- **ID kept.** Only frames whose box was given to the tracker are scored (otherwise 20% deletion would cap the score at 80%). Score: the share of those frames held by the clip's most common ID. Counts are summed over clips.
- **Gaps.** Each gap follows 2 s of detections, then repeats: settle, gap, settle, gap. Gap length in seconds is converted with each clip's frame rate. A gap is scored if the drone is labeled on the frame before it and on some frame before the next gap. "Held" gaps are the ones a track held the drone on the frame before; recovery is the share of held gaps where the same ID holds it on the first labeled frame after. Unheld gaps are reported but not scored, so a tracker that outputs nothing gets no rate rather than a flattering one.
- **Prediction error.** On each frame of a held gap where the drone is labeled: the held track's `predicted_box` from the frame before, against the labeled box.

## Why Tether loses the drone: the camera hunts

The labeled boxes are right. The drone sits inside them on every frame checked. But the drone's image motion is a sawtooth. The camera is a pan-tilt unit keeping the drone near its crosshair, so the drone drifts about 28 px a frame one way for 8–10 frames, then snaps back at about 50 px a frame as the camera re-centers. In one test clip (`20190925_134301_1_2`), the box goes 721 → 987 px over frames 0–11, then back to 724 by frame 17.

What a deletion does there:

1. The drone's box is deleted, so the track goes searching and coasts on its old velocity.
2. If the camera re-centers meanwhile, the prediction lands on the wrong side (IoU 0 with the drone), so the returning detection can't match it.
3. That detection starts a new tentative track, which needs 3 frames in a row to be confirmed. A deletion in either of its next 2 frames (a 36% chance at 20%) drops it, and the next detection starts yet another.

That clip used 60 IDs in 760 scored frames. Across the split, some track held the drone on 87.8% of scored frames, but the most common ID only held it on 26.1%. Tether keeps finding the drone; it keeps losing the identity.

Frame-to-frame prediction is not the problem. With nothing deleted, the track holding the drone predicted the next frame's box below IoU 0.3 on only 3.4% of 73,971 frames, about the same as a perfect constant-velocity guess from the labels (`2·box[f−1] − box[f−2]`, 3.9% of 79,436 frames) and better than "stay where it was" (6.3%). What breaks is coasting: a missed frame means predicting further ahead on a velocity the camera is about to reverse, and the gap errors above grow about linearly with gap length (33 → 73 → 142 px), as coasting on a fixed velocity would. The rules for new tracks then turn each loss into a run of fresh IDs. Those two (coasting through misses, and how tentative tracks confirm and drop) are where to look next. Not measured yet: how ByteTrack does on the same files, and how much a change to either would help.

## Reproduce

```
pip install -e .[demo]
tether-antiuav data/Anti-UAV300
```

`data/Anti-UAV300` is `Anti-UAV-RGBT.zip` from the [Anti-UAV repository](https://github.com/ZhaoJ9014/Anti-UAV) (5.6 GB; only the `visible.*` files and `label_new/` are needed). Defaults: `--split test --drop-rates 0.2 --gap-seconds 0.25 0.5 1.0 --seed 0`. Any tracker command can be scored with `--tracker` and `--name`, as for `tether-mot17`. Detection files are saved under `results/antiuav/detections/<condition>/`, so other trackers run on the very same input. Results go to `results/antiuav/trackers/<name>/results.json`. A run takes about 18 minutes, nearly all of it the tracker.
