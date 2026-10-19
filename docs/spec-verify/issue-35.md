# spec-verify: issue #35, capping the search gate by the distance to the nearest other track

Spec: the acceptance criteria in [issue #35](https://github.com/Khriis-K/Tether/issues/35), plus the decision that only confirmed tracks count toward the cap (the tentative track a missed detection starts is what the gate takes back, so counting it would undo #29's gain).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Mutations ran on a scratch copy of `python/` so the benchmark runs going at the same time kept importing the real code. Tests are in `python/tests/test_search_gate_cap.py`.

### Criterion C1: an isolated searching track re-matches within its full gate
Test: test_isolated_searching_track_rematches_within_its_full_gate
Mutation: with no other confirmed track, the nearest distance defaults to 0 instead of inf, so the cap shuts the gate.
Result: VERIFIED

### Criterion C2: a nearby track shrinks the gate to the cap
Test: test_nearby_track_shrinks_the_gate_to_the_cap
Mutation: `_capped_gate` returns the full gate, ignoring the cap.
Result: VERIFIED

### Criterion C2b: with the cap off (inf), the same detection re-matches
Test: test_without_the_cap_the_same_detection_rematches
Mutation: the tracker ignores the cap it was given and always uses 0.5.
Result: VERIFIED. The first mutation tried, deleting the `inf` shortcut in `_capped_gate`, survived. That mutation changes nothing (inf times any distance is still inf), so it says nothing about the test. This one does break the behavior.

### Criterion C2c: a detection inside the cap still re-matches
Test: test_detection_inside_the_cap_still_rematches
Mutation: the cap is halved again (`cap * nearest / 2`).
Result: VERIFIED

### Criterion C3: the cap measures to other tracks, never the track itself
Test: test_cap_measures_to_other_tracks_never_the_track_itself
Mutation: `other is not track` dropped, so the nearest distance is the track to itself, 0.
Result: VERIFIED

### Criterion C3b: tentative tracks don't cap the gate
Test: test_tentative_tracks_dont_cap_the_gate
Mutation: `other.confirmed` dropped, so the tentative track at the detection caps the gate.
Result: VERIFIED

### Criterion C4: the CLI passes --search-gate-cap to the tracker
Test: test_cli_passes_the_cap_to_the_tracker
Mutation: the CLI parses the flag but doesn't pass it to `Tracker`.
Result: VERIFIED

### Criterion C5: the gate is on by default
Test: test_search_gate.py::test_gate_is_on_by_default (replaces test_gate_is_off_by_default, which encoded the old default)
Mutation: the defaults reverted to off (`SEARCH_GATE = 0.0`, `SEARCH_GATE_GROWTH = 0.0`).
Result: VERIFIED. The first mutation tried, `SEARCH_GATE = 0.0` alone, survived. That mutation leaves the gate on (growth 0.25 still opens it around a detection at distance 0), so the default being on isn't what it breaks.

### Criterion C6: the cap is on by default
Test: test_cap_is_on_by_default
Mutation: default `SEARCH_GATE_CAP = math.inf`.
Result: VERIFIED

### Criterion C7: a much bigger or smaller detection is never a gate match
Test: test_search_gate_size.py::test_detection_much_bigger_or_smaller_starts_a_new_track (2x and 0.5x)
Mutation: the size check removed from `_by_search_gate`; then, separately, checked in one direction only (`b / a <= ratio`).
Result: VERIFIED (both mutations caught; the second only by the 0.5x case)

### Criterion C7b: a modest size change still re-matches
Test: test_search_gate_size.py::test_modest_size_change_still_rematches
Mutation: `SEARCH_GATE_MAX_SIZE_RATIO = 1.1`.
Result: VERIFIED

### Criterion C7c: size is compared to the last observed box, not the coasted prediction
Test: test_search_gate_size.py::test_size_is_compared_to_the_last_observed_box_not_the_coasted_prediction
Mutation: `_same_size(t.motion.box, b)` for `_same_size(t.box, b)`.
Result: VERIFIED. This test passes before the fix too, because without a size check there's nothing to compare. It guards how the check compares, so the mutation is what counts.

### Criterion C2-#29: a zero gate never matches (fixture changed)
Test: test_search_gate.py::test_zero_gate_never_matches_even_a_detection_centered_on_the_prediction
Mutation: a zero gate counted as on (`self._gate(t) >= 0`).
Result: VERIFIED. The fixture `CENTERED` was a 4x4 box on a 20x20 track. Once the size check went in, it would have made this test pass whatever the gate did, so it is now a narrow box of the same height (IoU still 0.2).

## Verification results
13/13 criteria VERIFIED.
Measurement criteria (cap fraction tuned on Anti-UAV val, MOT17 at both input sizes, how often the cap binds, the default decision): UNVALIDATABLE by this technique; reported in `docs/benchmarks/antiuav-search-gate-cap.md` for human review.
