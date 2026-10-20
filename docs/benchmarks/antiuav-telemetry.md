# Camera motion from the on-screen pan-tilt readout: the feasibility check

Run 2026-10-20, following up [antiuav-camera-motion.md](antiuav-camera-motion.md) (#32). Anti-UAV **val** (67 clips, 61,999 frames, 20 fps, 1920×1080). Nothing in the tracker changed. The question: the pan-tilt angles are burned into every frame ("方位" azimuth, "俯仰" elevation, to 0.01°), so can they stand in for the image-based camera motion that the sky was too blank to give?

**Result: the angles read reliably and convert to pixels consistently, but they can't compensate frame to frame.** The camera doesn't pan smoothly. It jumps about 550 px in roughly 5 frames, then holds still. The readout only updates about every 11 frames, so it knows *how far* the camera moved but not *when*. On the issue's sanity check, telemetry compensation makes the drone's frame-to-frame velocity variance worse, not better: 1454 → 6323 px² applied as a step, 1454 → 1586 px² (+9.1%) spread evenly between updates. Image-based compensation managed 1454 → 1370 (−5.8%).

## Step 1: reading the digits

Scratch scripts in `results/telemetry/` (not committed): `reader.py` builds the templates, `reader2.py` reads a frame, `read_all.py` runs a split, `analyze.py` does the cleanup and fits.

- **The text flips polarity.** The OSD draws white text over dark backgrounds and dark text over bright sky. So the reader works on an "ink" map: the absolute difference between the gray image and a 31 px median-blurred background. That map is bright for text either way.
- **Templates:** digits 0–9 and "." from 8 hand-read frames, matched by normalized cross-correlation. No OCR dependency.
- **Slots, not segmentation.** A first version split glyphs at blank columns. It failed wherever a pole or building edge touched the text, merging blobs or adding fake ones: up to 29% of frames on some clips. The digits are fixed-pitch and the layout never moves (the elevation label sits at x = 276 on every val frame where it was located confidently). So the reader now tries 1, 2 or 3 integer digits at fixed slot positions, each followed by "." and two decimals, and keeps the best-scoring hypothesis.
- **Cleanup:** a read counts as confident when its weakest glyph scores ≥ 0.3 and beats the runner-up by ≥ 0.1. A value is accepted only once two consecutive confident frames agree. Real values hold for about 11 frames, so a one-frame flicker can't be a real update. The accepted value is then held forward.

| | Value |
|---|---|
| Frames | 61,999 |
| Confident reads | 61,478 (99.2%) |
| Frames with a value after cleanup | 61,973 (99.96%) |
| Jumps over 3° between accepted values | 0 |
| By-eye audit, 20 random frames | 20 / 20 correct |
| By-eye audit, 20 of the 2,000 weakest-scoring frames | 16 correct, 4 stale |

The 4 stale ones in the audit were not misreads. A bright object hid part of the readout, the frame wasn't confident, and the previous value was held: for example the screen shows 278.30 and the held value is 278.34. So the held value can lag the screen while the text is hidden.

## Step 2: what the readout measures

- **Update cadence:** a median of **11 frames** between changes (0.55 s, about 1.8 Hz). 67.4% of the 3,055 intervals are 10–12 frames. Another 12.7% are 21–23 frames, which look like an update where the value didn't change at 0.01°.
- **Resolution:** 0.01°, and steps of 0.01° do occur. At the scale below, **0.01° is about 19 px**.
- **The camera moves in jerks.** On textured clips, the image-based estimate from #32 shows the camera holding still, then moving about 150 px/frame for roughly 5 frames (about 550 px), then holding again. That is the sawtooth: the gimbal steps toward the drone instead of following it smoothly.
- **Lag:** fitting image-based camera motion summed between readout updates, the best shift is 0 frames on 8 clips and ±1 on 3. But each reading is a sample-and-hold of the angle, so it can land anywhere from mid-jerk to about 10 frames after one. On average the readout isn't late. On any single frame it can be one whole jerk behind.

## Step 3: degrees to pixels

Between consecutive readout updates, image-based camera motion (from #32's estimator, frames where it didn't fall back) was regressed on the change in angle. Azimuth was scaled by cos(elevation). The fit is through the origin, run on the 11 val clips where #32 fell back on under 25% of frames and at least 10 update windows were usable.

| | Value |
|---|---|
| px/° horizontal, per clip | −1736 to −1948 (median −1916) |
| R² per clip | 0.89 to 0.98 |
| px/° vertical, clips with enough elevation change | +2001 (R² 0.99), +1900 (R² 0.97), +968 (R² 0.71) |

The sign means azimuth increases to the right and elevation increases upward. About 1900 px/° puts the horizontal field of view near 1°, a long telephoto. The scale holds steady across these 11 clips, so there's no sign of zoom changing between them. **Not checked:** the 56 untextured clips, which are exactly where telemetry would be needed. Fitting the scale there against the drone's labeled motion is confounded, because the camera moves *because* the drone does. I didn't run that fit. The third vertical fit is weak (that clip's elevation barely changes), so the variance check below uses the horizontal median for both axes.

## Step 4: the #32 sanity check

Same pairs as #32: 58,214 frame pairs where the drone is labeled in both frames. Variance is the sum over x and y of the drone center's frame-to-frame displacement, after subtracting the camera motion from telemetry. Scale is −1916 px/° horizontal and +1916 px/° vertical.

| Compensation | Velocity variance, px² | Change |
|---|---|---|
| None | 1454 | |
| Image-based (#32) | 1370 | −5.8% |
| Telemetry, whole step on the update frame (best shift, 0) | 6323 | +335% |
| Telemetry, linear between updates (best shift, 0) | 1586 | +9.1% |
| Telemetry, spread over the W frames ending at the update (best: W 11, shift +1) | 1488 | +2.3% |

The step version is worse on all 67 clips. The linear version lowers the variance on 42 of 67 clips, but by 25% or more on only 5, and it raises it a lot on others. On the textured clips the gap to image-based is large:

| Clip | Raw | Image-based | Telemetry, step | Telemetry, linear |
|---|---|---|---|---|
| 20190925_140917_1_6 | 1782 | 477 | 7955 | 1746 |
| 20190925_133630_1_9 | 692 | 128 | 4152 | 702 |
| 20190925_131530_1_3 | 976 | 292 | 5186 | 937 |
| 20190925_141417_1_1 | 1931 | 923 | 9169 | 1913 |
| 20190926_103046_1_2 | 175 | 57 | 1703 | 129 |

**Why.** A 550 px jerk spread over 5 frames, corrected by a 550 px step on one frame, leaves an error of about −400, +150, +150, +100 px even when the step is perfectly timed. That's worse than not correcting. The timing is uncertain by up to 10 frames anyway. Spreading the step over the update interval removes the spike but puts the motion on frames where the camera was still.

### Over longer windows

Against image-based camera motion on the textured clips, the telemetry's error doesn't grow with the window. It stays at about one jerk:

| Window, frames | Windows | Camera motion, RMS px | Telemetry error, RMS px | Median | 90th percentile |
|---|---|---|---|---|---|
| 1 | 9,483 | 27.9 | 70.5 | 1.8 | 28.9 |
| 5 | 8,690 | 96.7 | 145.9 | 4.9 | 221.6 |
| 11 | 7,902 | 163.5 | 166.6 | 11.8 | 315.0 |
| 22 | 6,752 | 265.1 | 163.2 | 14.0 | 302.1 |
| 44 | 5,539 | 480.4 | 152.7 | 18.9 | 268.5 |

The median error is small, so the readout is accurate when it's current. The 90th percentile is a jerk the readout hasn't caught up with yet. Only from about 22 frames (1.1 s) does the telemetry error fall below the camera motion it removes. Over 5 frames, a typical detection gap, it adds more error than it removes.

A caveat for longer windows: the labeled drone's *compensated* displacement over 44 frames has a much larger variance than its raw displacement (52,095 → 440,213 px²). That isn't only telemetry error. The camera keeps the drone near the center, so in the image the drone's motion stays bounded, while in world angles it keeps going. Velocity variance stops being a fair measure of "smoother" at long windows, which is why the table above compares against image motion instead.

## What this rules out, and what it doesn't

- **Rules out:** telemetry as a frame-by-frame replacement for image-based compensation on Anti-UAV visible. The camera jerks over about 5 frames and the readout samples every 11. That's a hard limit of the data, not of the reader.
- **Doesn't rule out:** telemetry over long gaps (a second or more), where its error stays near one jerk while the camera keeps moving. That might help re-find a drone after a long miss. It's untested, and only the textured clips have been checked.
- **Learned along the way:** the sawtooth is a step-and-hold gimbal, about 550 px per jerk. A tracker could treat a sudden shift of the drone's position by roughly that much as camera motion rather than drone motion. That's an idea, not a measurement.
