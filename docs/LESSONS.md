# Lessons

What we learned building Tether, written to be re-read: before an interview, when writing a portfolio page, or when coming back to the project after a break. Read the first section for a two-minute refresh, and read the stories when you want the detail behind a claim.

Definitions live in [CONCEPTS.md](CONCEPTS.md); this file links to them rather than repeating them. Numbers are as of 2026-10-20 and come from the write-ups linked in each story. Update this file when the numbers change, and never round a claim up.

**Contents**
- [The two-minute refresh](#the-two-minute-refresh): pitch, what's built, headline results, debugging playbook
- [Stories](#stories): the problems we hit and how we met them
- [What didn't work](#what-didnt-work)
- [Honest limitations](#honest-limitations)

---

## The two-minute refresh

### The pitch
Tether is a multi-object tracker built to keep stable IDs on fast-moving and temporarily hidden objects, the cases where trackers usually hand out new IDs. A YOLO detector finds objects in each frame, and a hand-written tracker links them over time with a Kalman motion model, staged Hungarian matching and a track lifecycle. It's built in two passes: a Python prototype to get the tracking logic right, then a C++ port on an NVIDIA GPU (TensorRT, CUDA frame prep, parallel stages) whose output has to match the prototype. Every claim is backed by a benchmark: MOT17 against ByteTrack for crowds, and Anti-UAV drone footage with detections deleted on purpose for robustness.

### What's built, what's next
**Built (Python):**
- The tracker: Kalman motion model, a five-stage matching cascade, a track lifecycle, ByteTrack's low-score pass, a capped search gate, and a duplicate-track merge.
- The YOLO26n ONNX detector, which reads its input size from the model.
- A webcam/video demo and per-step timing logs.
- A MOT17 evaluation through TrackEval, and a ByteTrack baseline on the same detections.
- An Anti-UAV dropout simulator with its own scoring.
- A single-target prior that knows there's one drone (`--single-target`), kept only as a diagnostic upper bound, never a default.
- 230 automated tests.

**Not built yet:** the C++ port, TensorRT, CUDA frame prep, the parallel pipeline, the HUD, and the Docker image. There are **no speed results yet**. Talk about these as the plan, not as achievements.

### Headline results

| Claim | Number | Say this with it |
|---|---|---|
| Matches ByteTrack on MOT17 at the benchmark input size (1280×736) | Tether 41.46 HOTA / 49.38 IDF1 / 732 ID switches; ByteTrack 40.86 / 48.63 / 800 | Measured on the same 7 videos we used to choose the settings, with no held-out split. The 0.60 HOTA margin is small. |
| Close to ByteTrack at the shipped input size (640) | Tether 33.1 / 37.6 / 529; ByteTrack 33.5 / 37.9 / 549 | Slightly behind on HOTA and IDF1, ahead on ID switches. |
| Fixing the matching order closed most of the gap | 1270 → 779 ID switches, 35.6 → 40.9 HOTA (1280×736) | Found by ablation, not by guessing. See [the story](#tether-made-470-more-id-switches-than-bytetrack-22-23). |
| Higher input resolution, not a bigger model, fixed detection | Detector recall 33.7% → 54.6% across the 7 videos, at similar precision (85.4% → 84.3%) | A 3× bigger model at 640 still missed small people. |
| Recovers the drone after gaps far more often | Anti-UAV **test** (run once, at the end): ID kept at 20% deletion 26.1% → 84.0%; 1 s gaps recovered 27.2% → 88.9%. Val, which we tuned on: 85.3% and 88.8%. | Detections are the labeled boxes, so this tests the tracker, not the detector. Gap recovery meets its 80% target. ID kept doesn't: the target is 95% of the reachable ceiling, 89.3% on test, so it's 5.3 points short. See `docs/benchmarks/antiuav-dropout.md`. |

### Debugging playbook
The habits these stories taught, in the order you'd reach for them:
1. **Check the ceiling upstream.** Before tuning a stage, compare its output to what its input allows. A tracker can't beat its detector. *([The first MOT17 score](#the-first-mot17-score-was-low-and-the-tracker-wasnt-to-blame-6))*
2. **Check your metric before you believe it.** Run the null case: what would a useless system score? And make a homemade metric follow the official rules. *([The faked "ID kept" score](#the-id-kept-score-could-be-faked-by-doing-nothing-8); [our own precision numbers](#our-own-precision-numbers-were-wrong))*
3. **Change one thing at a time.** When two systems differ, ablate. The obvious suspect is often wrong. *([Tether vs. ByteTrack](#tether-made-470-more-id-switches-than-bytetrack-22-23))*
4. **Compare each stage to its best possible version.** The stage that looks weak may already be at its limit, and the real loss may be somewhere else. An oracle that cheats on one stage tells you that stage's most. *([Losing the drone](#tether-found-the-drone-but-kept-losing-its-id-8-29); [the oracle](#a-perfect-re-association-would-get-most-of-the-way-not-all-34))*
5. **Test on a benchmark that can show the cost.** A change tuned on one dataset needs a second one where it can fail, checked per video, not just in total. *([Drones vs. crowds](#the-fix-for-drones-broke-crowds-29-35))*
6. **Judge by the outcome, not a stand-in.** A better intermediate number doesn't always mean a better result. *([Damping](#better-predictions-didnt-mean-more-re-matches-30); [fewer new IDs](#fewer-new-ids-same-score-31))*
7. **Write the ceiling next to the target.** If the best possible score is below the target, tuning can't get there. *([Two IDs at once](#the-drone-held-two-ids-at-once-36-33))*
8. **Count the failure on real data before fixing it.** A scripted scene shows a failure can happen, not how often. *([Stopping one swap](#stopping-one-swap-cost-more-than-it-saved-37))*
9. **Check that a quality gate passes for the right reason.** "Enough points" isn't "the right answer"; pair it with a check on the outcome. *([The camera-motion check](#the-camera-motion-check-said-fine-when-it-wasnt-32))*

---

## Stories

Roughly in the order we hit them. Each one opens with a two-sentence version to say out loud; the rest is detail for follow-up questions.

### A fast object got a new ID every frame (#2)
**In two sentences:** a new track has no velocity yet, so it couldn't follow an object that was already fast on its first frame, and it handed out a new ID every frame. A scripted test showed the failure, and a one-time center-distance match for brand-new tracks took it from 20 IDs to 1.

**Problem:** an object moving 50 px/frame from its very first frame got 20 IDs in 20 frames.

**How we found it:** issue #2 said to add a fallback only if a scripted fast-start test failed. It failed, and the cause was a chicken-and-egg problem: no velocity yet, so no overlap, so no match to learn a velocity from (the [Kalman filter](CONCEPTS.md#kalman-filter-motion-model) gotcha).

**What we did:** the [center-distance fallback](CONCEPTS.md#center-distance-fallback), plus [process noise](CONCEPTS.md#process-noise-how-fast-the-filter-adapts) tuning. 20 IDs became 1.

**Lesson:** write the hard scene as a test first, and add the fix only if the test fails. A scripted 20-frame scene found this in seconds, with no video or model, and kept the fix from being speculative. See ADR 0001.

### The first MOT17 score was low, and the tracker wasn't to blame (#6)
**In two sentences:** the tracker's recall sat right at the detector's own ceiling, so the detector was the bottleneck, not the tracker. Raising the input resolution from 640 to 1280 found far more of the small people, while a 3× bigger model at 640 found none of them.

**Problem:** HOTA 27.8 on the first MOT17 run.

**How we found it:** the tracker's recall (32.1%) sat right at the detector's own recall (about 34%). A tracker can't keep an ID on a person the detector never found. Splitting recall by person height showed the pattern: people under 100 px were almost never found.

**What we did:** tested the cheapest explanation, resolution, before anything else. 1920×1080 frames shrunk to 640 make people 3× smaller. At 1280 input, 50–100 px people on MOT17-04 went from 0% found to 61%. A 3× bigger model at 640 still found 0%. We then split the [benchmark detection set from the shipped detector](CONCEPTS.md#benchmark-detection-set-vs-shipped-detector), and later moved to a 1280×736 rectangle that costs 42% less than the square.

**Lesson:** before tuning a stage, check it isn't capped by the stage in front of it. Comparing a stage's output to its input's ceiling is a quick way to check. See ADR 0002.

### Our own precision numbers were wrong
**In two sentences:** our quick scoring script counted detections on people the official benchmark ignores as false alarms, which understated precision by 2–9 points. Rescoring with the official rule flipped one of our conclusions.

**Problem:** detector precision looked worse at higher resolution than it was.

**Cause:** our scoring script counted detections on static people, people on vehicles, and reflections as false alarms. MOT17's official rule ignores those.

**What we did:** rescored with the official rule, which is to match against *all* ground-truth boxes first, then drop matches to the ignored classes. Some conclusions held and one flipped: on MOT17-02, native resolution went from behind to ahead on precision.

**Lesson:** a quick homemade metric has to follow the official rules, or it measures something else. Check it against the official tool on one case before trusting it. See `docs/handoffs/2026-10-18-mot17-precision-ignore-classes.md`.

### Tether made ~470 more ID switches than ByteTrack (#22, #23)
**In two sentences:** the obvious suspect, ByteTrack's low-score pass, explained under a fifth of the gap, so we swapped in ByteTrack's behaviours one at a time. Only matching order mattered: new tentative tracks were stealing detections from confirmed ones, and matching confirmed tracks first cut ID switches from 1270 to 779.

**Problem:** on the same detections, Tether made 1270 ID switches to ByteTrack's 800.

**First guess, wrong:** ByteTrack's low-score second pass. A run of ByteTrack with that pass turned off showed it was worth only 0.2 HOTA, well under a fifth of the gap.

**How we found it:** [ablation](CONCEPTS.md#ablation). We swapped one ByteTrack behaviour into Tether at a time: gate, Kalman filter, confirmation frames, reported box, assignment, matching order. Only matching order moved the numbers. Tether matched tentative and confirmed tracks in one pass, so a tentative track could steal a confirmed track's detection. We then counted how often that happened, 847 times, which turned a correlation into a mechanism. One more clue fit: confirming after 2 frames instead of 3 made things *worse*, as it should if tentative tracks are the thieves.

**What we did:** a [matching cascade](CONCEPTS.md#matching-cascade-confirmed-tracks-first). ID switches went from 1270 to 779 and HOTA from 35.6 to 40.9. Adding the low-score pass afterward (#26) was a smaller, consistent gain.

**Lesson:** when two systems differ, change one thing at a time and measure each change. The obvious suspect was the wrong one. Once a candidate wins, count the failure it explains to confirm the mechanism.

### The "ID kept" score could be faked by doing nothing (#8)
**In two sentences:** with one drone per clip, a tracker that never drops its track would score a perfect "ID kept", even with its box nowhere near the drone. Asking what a useless tracker would score caught that, plus two more traps in the scoring rules.

**Problem:** Anti-UAV clips have one drone each. A tracker that never drops its one track would score 100% "same ID kept", even if the box drifted off the drone.

**What we did:** required IoU ≥ 0.5 with the labeled drone (the [null-case test](CONCEPTS.md#null-case-test) entry). The same question caught two more traps in the scoring rules. If deleted frames were scored, 20% deletion would cap the score at 80%. And if gaps a tracker never held were scored, outputting nothing would give a flattering recovery rate. Instead, unheld gaps get no rate at all.

**Lesson:** ask the null question of every scoring rule, not just the headline metric. One question found three traps here.

### Tether found the drone but kept losing its ID (#8, #29)
**In two sentences:** prediction was already as good as a perfect constant-velocity guess, so a better motion model wasn't the fix. The real problem was that one missed match lost the ID for good, and a search gate that widens while the track is lost doubled ID kept and nearly tripled recovery after 1 s gaps.

**Problem:** with nothing deleted, some track held the drone on 93% of frames, but the main ID held it on only 37.6%. One 1,000-frame clip used 81 IDs.

**How we found it:** the obvious move was a better motion model, but we measured prediction first. The tracker missed its next-frame prediction on 3.4% of frames. A *perfect* constant-velocity guess made from the labels missed on 3.9%. So prediction was already at its model's limit. The real loss was that every miss was permanent: the searching track could never re-match by IoU. (The misses themselves come from the camera, as the [Anti-UAV](CONCEPTS.md#anti-uav) gotcha explains.) We also ran ByteTrack on the same clips, and it was only 0.6–7 points better on every test, so the problem wasn't specific to Tether.

**What we did:** the [growing search gate](CONCEPTS.md#growing-search-gate). On val, ID kept at 20% deletion went from 30.5% to 61.1%, and 1 s gap recovery from 31.3% to 88.4%.

**Lesson:** compare each stage to its best possible version before choosing what to fix. The stage that looks weak (prediction) may already be at its limit, while the real loss is somewhere else (recovery).

### The fix for drones broke crowds (#29, #35)
**In two sentences:** a wide search gate can't grab the wrong object when there's only one drone, so Anti-UAV could never show its cost, but on MOT17's crowds it added hundreds of ID switches. Capping the gate at half the distance to the nearest other track fixed the crowds and kept most of the drone gain.

**Problem:** every gate setting that helped Anti-UAV hurt MOT17. The one we picked added 513 ID switches.

**Why one benchmark couldn't see it:** see the [growing search gate](CONCEPTS.md#growing-search-gate) gotcha. With one drone per clip, a false match is impossible, so only MOT17 could show the cost.

**What we did:** four variants that tweaked the gate's shape failed. The fix was to make the gate depend on the scene: the [nearest-track cap](CONCEPTS.md#nearest-track-cap). The total MOT17 numbers then looked fine, but checking per video showed MOT17-09 had lost 1.7 HOTA. That led to the size check (see the cap's second gotcha). The gate is now on by default. MOT17 ended slightly better than before (731 ID switches, down from 748), and Anti-UAV kept most of its gain (59.9% ID kept, 86.6% of 1 s gaps).

**Lesson:** a change tuned on one benchmark needs a second benchmark that can show its cost. Check per video too, because totals can hide one video getting worse.

### Better predictions didn't mean more re-matches (#30)
**In two sentences:** damping the velocity of a lost track lowered its average prediction error but made it re-find the object *less* often. A mean distance can't tell you whether a prediction landed close enough to match, so we judged by recovery and left damping off.

**Problem:** damping the velocity while coasting lowered the average prediction error, but without the gate it *lowered* recovery (0.5 s gaps: 54.4% → 44.1%).

**How we read it:** see the [damping](CONCEPTS.md#velocity-damping-while-coasting) gotcha. In short, a mean distance can't show whether a prediction landed close enough for IoU to match. The null version, freezing the box, mattered too: it tied the best damping at 1 s, which says the velocity estimate is useless past about half a second. It also settled one of the issue's options, zeroing the velocity after k frames: freezing outright did no better, so a cutoff had nothing to add.

**What we did:** kept damping in the code but off by default. With the gate and cap on, it adds ID switches on MOT17 at 640.

**Lesson:** judge a change by the outcome you care about, not a stand-in for it. Always run the null version.

### Experiments changed while they ran
**In two sentences:** editing the code during a long benchmark run silently changed the code that run was using, and parallel runs could overwrite each other's files. We froze baselines in a separate git worktree and gave every run its own folder.

**Problem:** the package is an editable install, so editing `python/tether` during a long benchmark run changed the code the run was using. Parallel runs sharing a results folder could also overwrite each other's files.

**What we did:** froze the baseline in a separate git worktree (`git worktree add ../Tether-baseline HEAD`), and gave every run its own `--work-dir`. We tune on Anti-UAV *val* and keep *test* for one final run, so the test numbers don't leak into the tuning (see [train/test leakage](CONCEPTS.md#traintest-leakage)).

**Lesson:** when you compare A and B, make sure each one stays exactly as it was until its run finishes.

### The drone held two IDs at once (#36, #33)
**In two sentences:** after a missed jump, the drone's detection started a new track that got confirmed while the old one was still searching, so the drone held two IDs that traded it back and forth. Merging the new track into the old one when it's confirmed took ID kept at 20% deletion from 59.9% to 85.3%, with every MOT17 video within 0.01 HOTA.

**Problem:** ID kept at 20% deletion was 59.9% on val, against a 95% target. We measured first: of 785 changes of which ID held the drone, 724 went to a track born while the old holder was still searching. Duplicates were nearly all of what was left.

**What we did:** the issue's first ideas needed the duplicate to be born near the old track's prediction, but it usually wasn't. The median was 5.6 box sizes away. So the [duplicate-track merge](CONCEPTS.md#duplicate-track-merge) got its own reach, much wider than the gate: 10 + 1 box sizes per missed frame. It's capped like the gate, so it can't reach a neighbor in a crowd. Changes of which ID held the drone fell from 913 / 785 to 29 / 37, and every MOT17 video stayed within 0.01 HOTA. See `docs/benchmarks/antiuav-duplicates.md`. The cost is in sparse scenes, where the wide reach can hand an ID to a newcomer. #37 tried to fix that and shelved it ([below](#stopping-one-swap-cost-more-than-it-saved-37)).

**Still open:** even a perfect tracker that gives up after 30 frames can reach only 94.0% on test at 20% deletion (94.2% with nothing deleted), because the drone sometimes leaves the labels for longer. On val the same [ceiling](CONCEPTS.md#reachable-ceiling) is 96.5%. Test only clears 95% with a 3 s search (97.2%), and deletion moves the ceiling by at most 0.2 points (`docs/benchmarks/antiuav-ceiling.md`). #33 kept the 30-frame search and made the target relative: ID kept ≥ 95% of the ceiling, 91.7% on val. A longer search would buy about 1.3 points of headroom on val at a cost in crowds, while the tracker itself was 11 points under the ceiling.

**Lesson:** write down the ceiling next to the target. If the ceiling is below the target, no amount of tuning will reach it. And measure what's actually losing the score before choosing the fix.

### Stopping one swap cost more than it saved (#37)
**In two sentences:** in a nearly empty scene, a person who walked out of frame could hand their ID to a newcomer, so we tried an "edge rule": a track last seen at the frame edge never gives its ID away. It fixed the scripted scene, but on real data the swap was rare and the rule cost about 1 point of Anti-UAV ID kept, so we shelved it.

**Problem:** the gate and the merge are measured in box sizes. Around a big, close-up box they reach across most of the frame, and with nobody else in view the [nearest-track cap](CONCEPTS.md#nearest-track-cap) never binds. A scripted webcam scene swapped IDs for every gap from 0 to 29 frames.

**How we found the cost:** we measured before deciding.
- **MOT17 ground truth:** only 23 of 194 risky disappear/appear pairs happen at the edge. The rest are mid-frame, where the rule can't help.
- **Real webcam clip:** the size check already stopped the swap at the demo's input size.
- **Anti-UAV val:** ID kept fell from 85.3% to 84.3% at 20% deletion.
- **MOT17-04 at 1280×736:** regressed, because the rule blocked two *correct* re-matches of people standing at the top edge. Adding "and moving out" fixed MOT17-04 only with a threshold read off those two samples.

See the [edge rule](CONCEPTS.md#edge-rule-tried-shelved) and `docs/benchmarks/edge-rule.md`.

**What we did:** didn't merge it. The code stays on a branch, and the swap is listed under the limitations below. Telling "the same person came back" from "someone new" needs [appearance re-ID](CONCEPTS.md#appearance-re-identification-re-id), which a motion rule can't supply.

**Lesson:** count how often a failure happens on real data before fixing it. A scripted scene shows that a failure can happen, not how often. And a fix that trades a rare error for a common one is a loss, even when its own test passes.

### Fewer new IDs, same score (#31)
**In two sentences:** at 20% deletion, hundreds of new tracks died just because one of their first boxes was deleted, so we let a new track survive one miss. It cut the IDs handed out by a fifth, but ID kept didn't move, and MOT17 at 640 got slightly worse, so we shelved it.

**Problem:** a new track needed 3 hits in a row. At 20% deletion, one deleted box in its first frames dropped it, and the next detection started yet another ID.

**How we found the size of it:** we measured first. On Anti-UAV val at 20% deletion, 637 new tracks died only because their box was deleted, against 15 confirmed tracks that expired. By the issue's own test, that was big enough to build.

**What we did:** a [leaky counter](CONCEPTS.md#leaky-counter-tried-shelved): a new track survives one miss and is confirmed on hits minus misses. New IDs fell from 1925 to 1508, but ID kept went from 85.3% to 85.2%. Two things ate the gain. A miss costs a whole hit, so surviving one saves at most a frame. And the [duplicate-track merge](CONCEPTS.md#duplicate-track-merge) was already giving the drone its old ID back, so the saved IDs were ones nothing scored. MOT17 at 640 lost 0.46 HOTA. We didn't find why, and false tracks barely changed (93 → 95). See `docs/benchmarks/tentative-miss.md`.

**Lesson:** a count that goes down isn't a score that goes up. Before building, ask whether the thing you're counting actually costs the metric, here whether a dying new track cost an ID that was scored. Also, mutation-testing a file in place while benchmarks import it can change what the benchmarks run, so benchmark a frozen copy.

### The camera-motion check said "fine" when it wasn't (#32)
**In two sentences:** before building camera-motion compensation for Anti-UAV, we counted how many frames had enough background points to estimate the camera's motion, and the first count said all of them. The points were flat sky and crosshair lines, all reporting "no motion"; done properly, 76.2% of val frames have nothing to register against, so we closed the issue without building it.

**Problem:** the drone's image motion is mostly the pan-tilt camera chasing it. Compensating for camera motion would remove the cause instead of working around it, but only if the mostly empty sky has enough texture to measure the motion from.

**How we found it:** the first prototype reported 0% fallback with hundreds of inliers per frame, while estimating 0.1 px/frame of camera motion against a drone moving about 20 px/frame. The on-screen azimuth readout showed the camera panning. Flat sky has no gradient, so optical flow reports zero motion there and passes its own consistency check: the [aperture problem](CONCEPTS.md#camera-motion-compensation-gmc-tried-not-feasible). With a texture filter, the survivors were all next to the burned-in crosshair, which never moves either. Masking both gave the real number. The drone's velocity variance, the issue's sanity check, confirmed it: down only 5.8% over all of val, but halved on the few textured clips. See `docs/benchmarks/antiuav-camera-motion.md`.

**What we did:** closed #32 at the feasibility step, as the issue planned. Along the way we found the pan-tilt angles burned into every frame, which the research had looked for in the dataset's files and not found. That's a possible source of camera motion that doesn't need texture. A follow-up read it: the angles are reliable, but they update every 11 frames while the camera jumps in 5-frame jerks, so they can't compensate frame by frame either. See `docs/benchmarks/antiuav-telemetry.md`.

**Lesson:** a quality gate can pass for the wrong reason. A fallback rate counts how many points the estimate used, not whether it was right, so pair it with a check on the outcome. And look at the data before the research summary: the telemetry the research ruled out was on screen the whole time.

### A perfect re-association would get most of the way, not all (#34)
**In two sentences:** before porting to C++, we wanted to know whether more Python work on re-finding the drone was worth it, so we built an oracle that knows each clip has one drone and lets a lost track take the detection anywhere. It recovers 7.9 of the 11.2 points between Tether and the reachable ceiling (85.3% → 93.3% on val at 20% deletion); most of the rest is the drone jumping away from a track that was still tracked.

**Problem:** ID kept at 20% deletion was 85.3% on val, against a 96.5% ceiling and a 91.7% target. We didn't know how much of that gap was re-association, the part more Python rules could still fix, and how much was something else.

**How we found it:** an [oracle experiment](CONCEPTS.md#oracle-experiment-upper-bound), `--single-target`, off by default: when the only confirmed track is searching, it takes the nearest detection anywhere. Then every frame between the score and the ceiling went into one bucket. For the default, another ID holding the drone is 4.66 points and a searching track not getting it back is 3.53. The oracle takes both to zero. What's left with the oracle is jumps (2.80 points: the track was tracked last frame, misses the IoU match, and gets the drone back a frame later) and each clip's first track being confirmed (0.29). The first run crashed on a coasted box that had turned inside out, with no size left to measure distance in. A test now pins that down. See `docs/benchmarks/antiuav-single-target.md`.

**What we did:** kept the option as a labeled diagnostic. Its numbers never count toward #28, and it won't be ported. On MOT17 its condition held on 0 of 5,316 frames, so it changed nothing there. To meet the target by re-association alone, a general rule would have to recover 79.9% of what an oracle that knows the answer recovers.

**Lesson:** before spending more effort on a stage, cheat on it once and measure. The oracle sets the most that stage can give, and what it leaves behind points at the next problem. Here that's motion between frames, not re-finding the drone.

---

## What didn't work

Each of these was measured, not guessed, and the write-ups have the tables. Interviewers often ask about dead ends.

- **The low-score pass as the cause of the ByteTrack gap.** It was worth 0.2 HOTA, under a fifth of the gap. It was still worth adding later as a small, consistent gain (#26). See `docs/benchmarks/mot17-bytetrack.md`.
- **Six other ByteTrack behaviours, one at a time (#22):**
  - Its score-fused match gate made Tether worse, because YOLO26's low scores make the gate stricter.
  - Lowest-total-cost assignment changed almost nothing.
  - Its Kalman filter helped HOTA and IDF1 but added 172 ID switches on top of the new matching order. Why is still unknown.
  - Pedestrian process noise (1/160) made ID switches worse.
  - Confirming after 2 frames instead of 3 made ID switches worse.
  - Reporting the filtered box instead of the detection changed nothing.

  See `docs/benchmarks/mot17-association-ablations.md`.
- **OC-SORT's three add-ons (#25).** OCR's gains swung by video (+2.5 HOTA on MOT17-02, −2.1 on MOT17-09), OCM was slightly negative, and ORU was near zero. None went into the tracker. See `docs/benchmarks/mot17-lowscore-ocsort-ablations.md`.
- **Reshaping the search gate (#29).** We tried the gate on tracks tracked last frame too, the gate after tentative tracks, distance per axis, and a gate scaled by speed. None fixed the MOT17 cost. The speed-scaled one kept MOT17 flat but lost most of the Anti-UAV gain. See `docs/benchmarks/antiuav-search-gate.md`.
- **Velocity damping (#30).** It's in the code but off by default: without the gate it hurts recovery, and with the cap it adds ID switches at 640. See `docs/benchmarks/antiuav-coast-damping.md`.
- **The edge rule (#37).** Blocking the gate and merge for tracks last seen at the frame edge fixed a scripted sparse-scene swap. It also cost Anti-UAV val about 1 point of ID kept (85.3% → 84.3% at 20% deletion) and regressed MOT17-04 at 1280×736. A velocity-gated version fixed MOT17-04 only with a threshold fit to two samples. See `docs/benchmarks/edge-rule.md`.
- **Letting new tracks survive a miss (#31).** A leaky counter (survive one miss, confirm on hits minus misses) cut new IDs on Anti-UAV val from 1925 to 1508 at 20% deletion, but ID kept went from 85.3% to 85.2%. MOT17 at 640 lost 0.46 HOTA (33.13 → 32.67) for a reason we didn't find. See `docs/benchmarks/tentative-miss.md`.
- **Camera-motion compensation from the image (#32).** Closed at the feasibility check, not built. On Anti-UAV val, 76.2% of frame pairs had too few textured background points to estimate the camera's motion, and compensation lowered the drone's velocity variance by only 5.8%. Looser thresholds cut the fallback rate but not the variance. See `docs/benchmarks/antiuav-camera-motion.md`.
- **Camera motion from the on-screen pan-tilt readout (#32 follow-up).** The angles read on 99.2% of val frames with no implausible jumps, and convert at a steady ~1900 px/° on the textured clips. But the camera moves in ~550 px jerks over about 5 frames and the readout updates only every ~11, so it knows how far the camera moved, not when. The drone's velocity variance got worse: 1454 → 1586 px² (+9.1%) spread between updates, 1454 → 6323 as a step. See `docs/benchmarks/antiuav-telemetry.md`.
- **Bigger detector models at 640.** YOLO26m found 0% of 50–100 px people on MOT17-04, the same as the nano model. See ADR 0002.

---

## Honest limitations

Know these before someone asks.

- **MOT17 results are on tuning data.** Every MOT17 number comes from the 7 train videos, and the same videos were used to choose the settings. There's no held-out split, so "beats ByteTrack" is a weak claim; "matches ByteTrack" is the honest one. We can't use a detector trained on MOT17 either, because that would leak (see [train/test leakage](CONCEPTS.md#traintest-leakage)).
- **The ByteTrack baseline isn't tuned to these detections either.** It's Ultralytics' BYTETracker with Ultralytics' defaults for YOLO detectors, not the paper's MOT17 settings, which assumed a detector trained on MOT17. Both trackers read the same detections, so the comparison is fair, but neither is at its best possible score.
- **At 640, Tether is slightly behind ByteTrack** on HOTA (0.4) and IDF1 (0.3). The cause hasn't been found.
- **Anti-UAV is short of its target.** ID kept at 20% deletion is 84.0% on test (the one final run) against a target of 95% of the reachable ceiling (94.0%, so 89.3%), and 85.3% on val against 91.7%. Gap recovery meets its target (88.9% of 1 s gaps on test). The target is relative because even a perfect tracker with a 30-frame search could reach only 94.0% on test. The 5% allowance is carried over from the original target, not derived from anything measured. See `docs/benchmarks/antiuav-ceiling.md` (#33, #36).
- **Sparse scenes can swap IDs.** With nobody else in view, a person who walks out can hand their ID to a same-size newcomer within about a second. The gate and merge reach far around a big box, and the cap only binds in a crowd. The size check catches some cases, and on MOT17 most risky pairs are mid-frame, not at the edge. Fixing it properly needs appearance re-ID (#37, `docs/benchmarks/edge-rule.md`).
- **Most of the Anti-UAV gap is re-association, and closing it generally is hard.** An oracle that knows there's one drone reaches 93.3% at 20% deletion on val (an upper bound, not a result). A general rule would need 79.9% of that gain to hit the target. The 2.8 points even the oracle leaves are frames where the drone jumps away from a tracked track (#34, `docs/benchmarks/antiuav-single-target.md`).
- **Anti-UAV tests the tracker, not the whole system.** Detections are the labeled boxes with deletions added, not a real detector's output.
- **No speed claims yet.** Everything so far runs in Python on CPU ONNX Runtime. The C++, TensorRT and CUDA work, and its before-and-after numbers, hasn't been built yet.
