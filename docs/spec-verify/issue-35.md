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

### Criterion C5: the gate is on by default (MOT17 held at both input sizes)
Test: test_search_gate.py::test_gate_is_on_by_default (replaces test_gate_is_off_by_default, which encoded the old default)
Mutation: the defaults reverted to off (`SEARCH_GATE = 0.0`, `SEARCH_GATE_GROWTH = 0.0`).
Result: VERIFIED. The first mutation tried, `SEARCH_GATE = 0.0` alone, survived. That mutation leaves the gate on (growth 0.25 still opens it around a detection at distance 0), so the default being on isn't what it breaks.

### Criterion C6: the cap is on by default
Test: test_cap_is_on_by_default
Mutation: default `SEARCH_GATE_CAP = math.inf`.
Result: VERIFIED

## Verification results
9/9 criteria VERIFIED.
