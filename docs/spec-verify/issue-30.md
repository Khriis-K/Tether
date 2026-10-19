# spec-verify: issue #30, velocity damping while a track coasts

Spec: the acceptance criteria in [issue #30](https://github.com/Khriis-K/Tether/issues/30), plus the decision to land damping off by default (it only helps Anti-UAV together with the #29 gate, which is itself off by default; see `docs/benchmarks/antiuav-coast-damping.md`).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_coast_damping.py`.

### Criterion C1: velocity shrinks each coasted frame
Test: test_velocity_shrinks_by_the_damping_each_coasted_frame (damping 0.5 and 0.8)
Mutation: damping applied on the first coasted frame only (`frames_since_seen == 1`, `tracker.py`).
Result: VERIFIED

### Criterion C1b: the first coasted step is damped too
Test: test_first_coasted_step_is_damped_too
Mutation: the velocity is damped after the step instead of before (`D @ F` for `F @ D`, `kalman.py`), so the first coasted step runs at full velocity.
Result: VERIFIED

### Criterion C1c: damping 0 is the constant-position null
Test: test_zero_damping_holds_the_coasted_position_still
Mutation: 0 is treated as unset (`coast_damping or 1.0`).
Result: VERIFIED

### Criterion C2: velocity is unchanged while the track is matched
Test: test_matched_frames_are_not_damped
Mutation: every surviving track's prediction is damped, matched or not.
Result: VERIFIED

### Criterion C3: normal behavior resumes once the track matches again
Test: test_velocity_recovers_once_the_track_matches_again
Mutation: damping sticks once a track has ever missed.
Result: VERIFIED. The first version asserted that the last predicted step was about SPEED and survived the mutation (VACUOUS): a filter that keeps damping still advances SPEED per frame in steady state, just a constant distance behind. It now asserts that the prediction lands where the object is on the next frame, which the lag breaks.

### Criterion C4: damping is off by default
Test: test_default_is_undamped
Mutation: default `COAST_DAMPING = 0.9`.
Result: VERIFIED

### Criterion C5: the CLI passes --coast-damping to the tracker
Test: test_cli_passes_the_damping_to_the_tracker
Mutation: the CLI parses the flag but doesn't pass it to `Tracker`.
Result: VERIFIED

### Measurement criteria (prediction-error plot with the constant-position null, with and without the #29 gate, val ID kept and gap recovery vs. targets, MOT17)
These are benchmark runs, not behavior behind a function, and are reported in `docs/benchmarks/antiuav-coast-damping.md`. MOT17 non-regression with the defaults was checked directly: track files byte-identical to main's on all 81 files compared.
Result: UNVALIDATABLE (not mutation-testable)

## Verification results
7/7 behavior criteria VERIFIED.
Measurement criteria: UNVALIDATABLE by this technique; reported in the benchmark doc for human review.
