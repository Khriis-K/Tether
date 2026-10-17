# Center-distance fallback for once-seen tracks

Matching is plain IoU between detections and each track's Kalman-predicted box. One case fails: an object already moving fast on its first frame. A new track has no velocity yet, so its prediction is its last box, which doesn't overlap the next detection, so it never gets the match that would teach it a velocity. Issue #2 said to add a center-distance fallback only if the fast-start test failed. It did.

**Decision:** after IoU matching, any track seen exactly once that is still unmatched gets a second Hungarian pass against the still-unmatched detections, by center distance measured in track box sizes (valid up to `MAX_CENTER_DISTANCE = 4.0`). Tracks with two or more hits never use it: they have a velocity, and the prediction is their job.

## Before and after

Scripted scenes from `python/tests/test_motion.py`, 20×20 boxes, counted as distinct IDs given to one object (1 is correct):

| Scene | IoU on predicted box only | With fallback |
|---|---|---|
| Fast start: 50 px/frame from frame 0, 20 frames | 20 (new ID every frame) | 1 |
| Ramp: 2 → 60 px/frame, then holds 60, 50 frames | 1 | 1 |

The ramp passes without the fallback, which is what shows predicted-box matching carries sustained fast motion by itself.

## Consequences

- A detection up to 4 box sizes from a once-seen track takes over its ID. If a one-frame false detection is followed by a real object close by, the real object inherits the false one's ID. Accepted: one-frame tracks are about to become "tentative" in the lifecycle work anyway.
- `MAX_CENTER_DISTANCE = 4.0` is a starting guess, not a measured value. Tune it on Anti-UAV.
- An object that has been slow for a while and then jumps suddenly (a step change, not a ramp) still loses its ID: it has a velocity, so it doesn't get the fallback, and its prediction lands in the wrong place. Not covered by this decision.
- Velocity noise matters as much as the fallback. With DeepSORT's usual `1/160`, the ramp scene gives 4 IDs even with the fallback; `1/20` passes but sits at the edge, so we use `1/10`.
