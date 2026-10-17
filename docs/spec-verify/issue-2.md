# spec-verify: issue #2, Kalman motion model and predicted-box matching

Spec: the acceptance criteria in [issue #2](https://github.com/Khriis-K/Tether/issues/2) and SPEC.md user stories 7, 23, 27, 28. Risks for D1–D2 come from [ADR 0001](../adr/0001-center-distance-fallback-for-new-tracks.md).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and content-compared before the next.

### Criterion C1: every track has a Kalman filter predicting position and velocity
Given a steadily moving object When the tracker runs Then its predicted box lands on the next frame's box.

Test: python/tests/test_motion.py::test_predicted_box_leads_a_steadily_moving_object_to_its_next_box
Mutation: tracker.py, drop the per-frame `motion.predict()` loop (filter never steps forward).
Result: VERIFIED

### Criterion C2: matching compares detections against the predicted box
Given an object that speeds up until its boxes stop overlapping When the tracker runs Then it keeps one ID.

Test: python/tests/test_motion.py::test_object_jumping_far_every_frame_keeps_the_same_id
Mutation: tracker.py, IoU matching uses each track's last seen box instead of `motion.box`.
Result: VERIFIED

### Criterion C3: each track in the output includes its predicted next box
Given a steadily moving object When its track is reported Then `predicted_box` is the next frame's box, not the current one.

Test: python/tests/test_motion.py::test_predicted_box_leads_a_steadily_moving_object_to_its_next_box
Mutation: tracker.py, snapshot reports `predicted_box=self.box`.
Result: VERIFIED

### Criterion C4: an object that jumps far every frame keeps the same ID
Given boxes that don't overlap between frames (60 px steps, 20 px boxes) When the tracker runs Then one ID throughout.

Test: python/tests/test_motion.py::test_object_jumping_far_every_frame_keeps_the_same_id
Mutation: kalman.py, `VELOCITY_NOISE` set to the pedestrian value `1/160` (prediction lags a speeding-up object).
Result: VERIFIED

### Criterion C5: an object already moving fast on its first frame keeps the same ID
Given 50 px/frame from frame 0, no velocity history When the tracker runs Then one ID throughout.
[Implementation decision paired: center-distance fallback for once-seen tracks; velocity unknown at birth]

Test: python/tests/test_motion.py::test_object_already_fast_on_its_first_frame_keeps_the_same_id
Mutation a: tracker.py, remove the center-distance fallback pass.
Result: VERIFIED
Mutation b: kalman.py, `_INITIAL_VELOCITY_NOISE` set to `1/16` (new track is confident it's still, so the first match barely teaches velocity).
Result: VERIFIED

### Criterion C6: tests check behavior from the outside, not internal filter values
Not a code behavior, so no mutation exists for it. Evidence for review: `test_motion.py` imports only `Detection` and `Tracker`; nothing imports `tether.kalman`. `test_matching.py` tests `matching`'s public functions, as before.
Result: [UNVALIDATABLE: a property of the tests themselves; needs a human read]

### Criterion C7: output stays deterministic
Given the same detection file When the CLI runs twice Then the output bytes are identical.

Test: python/tests/test_cli.py::test_same_input_run_twice_gives_byte_identical_output
Mutation: kalman.py, seed new tracks' velocity with unseeded `np.random.normal` noise.
Result: VERIFIED

### Decision D1: the fallback applies only to once-seen tracks
Given a still object seen for 5 frames When the only detection is 3 box sizes away Then it gets a new ID.

Test: python/tests/test_motion.py::test_established_track_does_not_take_a_nearby_non_overlapping_detection
Mutation: tracker.py, fallback offered to every unmatched track (`hits == 1` check removed).
Result: VERIFIED

### Decision D2: the fallback is gated by distance in track box sizes
Given a once-seen track When the next detection is 25 box sizes away Then it gets a new ID.

Test: python/tests/test_motion.py::test_far_away_detection_after_a_new_track_gets_a_new_id
Mutation: matching.py, gate removed (`dists <= np.inf`).
Result: VERIFIED

Test: python/tests/test_matching.py::test_center_distance_uses_the_track_box_larger_side
Mutation: matching.py, box size taken from the smaller side.
Result: VERIFIED

## Verification results
9/10 criteria VERIFIED or HUMAN-VERIFIED.
Remaining: C6 (UNVALIDATABLE, needs human sign-off).
