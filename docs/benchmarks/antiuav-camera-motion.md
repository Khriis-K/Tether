# Camera-motion compensation: the feasibility check

Run 2026-10-20 for #32. Anti-UAV **val** (67 clips, 20 fps, 1920×1080). Nothing in the tracker changed: this is the issue's first step, which asks whether the sky has enough texture to estimate camera motion at all.

**Result: not feasible, closed before building.** On 76.2% of val frame pairs (47,194 of 61,932), too few points survive to fit the camera's motion, so the tool would fall back to identity. 39 of 67 clips fall back on every frame. Over all of val, compensation lowers the labeled drone's frame-to-frame velocity variance by only 5.8% (1454 → 1370 px²). Where the sky does have texture, the estimate works: the drone's velocity variance drops by half or more. There just isn't enough of that sky.

## The estimator

Scratch prototype `results/i32/feasibility.py` (not committed), the method from shortlist item #4 of [camera-motion.md](../research/camera-motion.md):

- Frames downscaled to 960×540 and converted to gray.
- Grid keypoints every 16 px, so flat regions get points too, unlike corner detectors.
- Masked out: the labeled drone box (padded by half its size) on both frames, the burned-in crosshair, and the on-screen text (top left and top right). The mask is shrunk by the LK window radius, so no window overlaps the overlay.
- Pyramidal LK optical flow (21×21 window, 3 levels), kept only if the forward-backward error is under 1 px and LK's minimum eigenvalue is at least 1e-3 (the point has texture to track).
- RANSAC partial affine (`cv2.estimateAffinePartial2D`, 1 px threshold). A frame "falls back" when fewer than 20 inliers survive.

The sanity check from the issue: for each pair of frames where the drone is labeled in both, the drone center's displacement before compensation, and after (its position minus where the camera motion alone would have moved its previous position). Variance is the sum over x and y.

## Two ways the first version lied

Both are worth knowing because they make the fallback rate look fine when it isn't.

1. **Flat sky "tracks" at zero motion.** The first run had no texture filter and reported **0% fallback** with 243 median inliers on 3 clips, but the estimated camera motion averaged 0.1 px/frame while the drone moved about 20 px/frame. On the same clips the on-screen azimuth readout was falling by about 0.5° over 35 frames, so the camera was panning. A grid point on uniform sky has no gradient, so LK leaves it where it is, the backward pass agrees, and RANSAC happily fits identity. That's the [aperture problem](../CONCEPTS.md#camera-motion-compensation-gmc-tried-not-feasible).
2. **The overlay sits just outside the mask.** With a texture filter, about 50–60 points per frame survived even on blank sky. Every one of them was within an LK window of a crosshair line. Those points are pinned to the screen, so they also vote for identity. Shrinking the mask by the window radius removed them.

## Results

| | Value |
|---|---|
| Frame pairs | 61,932 |
| Fall back to identity | 47,194 (76.2%) |
| Clips falling back on every frame | 39 of 67 |
| Inliers per frame pair, 10th / 50th / 90th percentile | 0 / 0 / 52 |
| Drone velocity variance, raw → compensated (58,214 pairs) | 1454 → 1370 px² (−5.8%) |
| Clips where the variance drops at least 25% | 11 of 67 |

By how often a clip falls back:

| Clip falls back on | Clips | Median compensated / raw variance |
|---|---|---|
| under 25% of frames | 12 | 0.48 |
| 25–75% | 9 | 0.89 |
| 75% or more | 46 | 1.00 |

The textured clips show the estimator isn't the problem. For example, `20190925_140917_1_6` falls back on 3.6% of frames and its drone variance goes 1782 → 477; `20190925_133630_1_9` falls back on 2.5% and goes 692 → 128.

### Other thresholds don't rescue it

On every 8th val clip (9 clips, 8,852 pairs):

| Min eigenvalue | Min inliers | Fall back | Velocity variance, raw → compensated |
|---|---|---|---|
| 1e-3 (as above) | 20 | 82.2% | 1478 → 1412 (−4.5%) |
| 3e-4 | 10 | 29.7% | 1478 → 1396 (−5.5%) |
| 1e-4 | 8 | 1.2% | 1478 → 1428 (−3.4%) |

Loosening takes the fallback rate from 82% to about 1%, but the variance drop stays within 3–6% throughout, and the loosest setting does worst: the extra points are flat sky again, voting for no motion. **The fallback rate alone can't show feasibility. The velocity-variance check has to go with it.**

## What this rules out, and what it doesn't

- **Rules out:** image-based compensation from the background as the fix for Anti-UAV visible. About three quarters of the frames have nothing to register against, and they're the plain-sky clips where the camera motion matters most.
- **Doesn't rule out: telemetry.** Every frame has the pan-tilt angles burned in at the top left ("方位" azimuth, "俯仰" elevation, to 0.01°). [camera-motion.md](../research/camera-motion.md) §1c looked for telemetry in the dataset's files and found none; it's in the pixels instead. It updates in steps every few frames, and turning degrees into pixels needs the focal length, which may change with zoom. Not tried here.
- **Caveat:** the mask uses the *labeled* drone box, which the real tool wouldn't have on frames where the detector misses. That only matters on textured frames, so it doesn't change the verdict.
