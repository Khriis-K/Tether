# spec-verify: issue #29, growing search gate for searching tracks

Spec: the acceptance criteria in [issue #29](https://github.com/Khriis-K/Tether/issues/29), plus the decision to land the gate off by default (MOT17 regressed under every setting that helped; see `docs/benchmarks/antiuav-search-gate.md`).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_search_gate.py`.

### Criterion C1: a searching track re-matches a detection outside IoU range but inside the gate
Test: test_searching_track_rematches_a_detection_outside_iou_range_but_inside_the_gate
Mutation: the search-gate stage selects no tracks (`tracker.py`).
Result: VERIFIED

### Criterion C2: the gate widens with frames_since_seen
Test: test_gate_widens_with_frames_since_seen
Mutation: the gate ignores frames_since_seen (`search_gate + search_gate_growth`, a fixed one-frame gate).
Result: VERIFIED

### Criterion C3: a detection outside the gate still starts a new track
Test: test_detection_outside_the_gate_starts_a_new_track
Mutation: the stage drops its per-track gate and falls back to `match_by_distance`'s default (4 box sizes).
Result: VERIFIED

### Criterion C4: the searching track gets its detection back from the tentative track the miss started
Test: test_searching_track_takes_its_detection_back_from_the_tentative_track_it_started
Mutation: the search-gate stage runs after the tentative IoU stage.
Result: VERIFIED

### Criterion C5: the gate is off by default, so tracks match main's
Test: test_gate_is_off_by_default
Mutation a: default `SEARCH_GATE = 0.1`. The first version of this test used a detection 1.25 box sizes away and survived a default gate of 1.0 (VACUOUS). It now uses a detection centered on the prediction, which any gate above 0 takes.
Result: VERIFIED
Mutation b: default `SEARCH_GATE_GROWTH = 0.05`.
Result: VERIFIED

### Criterion C6: a zero gate matches nothing, even at distance 0
Test: test_zero_gate_never_matches_even_a_detection_centered_on_the_prediction
Mutation: the stage no longer skips tracks whose gate is 0 (`distance <= 0` then matches a concentric box).
Result: VERIFIED

### Criterion C7: tether-track's --search-gate and --search-gate-growth reach the tracker
Test: test_cli_passes_the_gate_settings_to_the_tracker
Mutation: `cli.py` passes the default `SEARCH_GATE` instead of `--search-gate`.
Result: VERIFIED

### Criterion C8: match_by_distance applies each track its own gate
Test: test_match_by_distance_applies_each_track_its_own_gate
Mutation: one gate for every track, the largest (`matching.py`).
Result: VERIFIED

### Measurement criteria (val baseline, ByteTrack, tuning, false-match check, MOT17, distance to SPEC targets)
These are benchmark runs, not behavior behind a function, and are reported in `docs/benchmarks/antiuav-search-gate.md`. MOT17 non-regression with the defaults was checked directly: track files byte-identical to main's on all 81 files compared. The false-match check is zero by construction on Anti-UAV (the only detection is the labeled box), which the benchmark doc explains.
Result: UNVALIDATABLE (not mutation-testable)

## Verification results
8/8 behavior criteria VERIFIED.
Measurement criteria: UNVALIDATABLE by this technique; reported in the benchmark doc for human review.
