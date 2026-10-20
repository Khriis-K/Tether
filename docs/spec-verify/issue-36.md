# spec-verify: issue #36, merging duplicate tracks

Spec: the acceptance criteria in [issue #36](https://github.com/Khriis-K/Tether/issues/36), plus the design chosen after measuring (see [antiuav-duplicates.md](../benchmarks/antiuav-duplicates.md)): merge at confirmation with its own reach (`merge_reach + merge_reach_growth · frames_since_seen`, 10 + 1 by default), capped like the gate, same size by `SEARCH_GATE_MAX_SIZE_RATIO`, on by default.
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Mutations ran on a scratch copy of `python/` so the benchmark runs going at the same time kept importing the real code. Tests are in `python/tests/test_merge.py`.

### Criterion C1: a duplicate born while the original searches ends with the original's ID
Test: test_duplicate_born_while_the_original_searches_ends_with_the_original_id
Mutation: `_merge_duplicates` never merges (`if reachable:` → `if False:`).
Result: VERIFIED

### Criterion C1b: the merged track keeps following the object
Test: test_merged_track_keeps_following_the_object
Mutation: `take_over` keeps the old track's coasted filter instead of the new track's.
Result: VERIFIED

### Criterion C2: two objects side by side, both tracked the whole time, are never merged
Test: test_two_objects_side_by_side_tracked_the_whole_time_are_never_merged
Mutation: tracks matched this frame are merge candidates too.
Result: VERIFIED

### Criterion C3: a much bigger or smaller box is never merged
Test: test_a_much_bigger_or_smaller_box_is_never_merged (both directions)
Mutation: the `_same_size` check is dropped from `_is_duplicate_of`.
Result: VERIFIED

### Criterion C3b: a box within SEARCH_GATE_MAX_SIZE_RATIO is still merged
Test: test_a_slightly_different_size_is_still_merged
Mutation: heights must be exactly equal instead of within the ratio.
Result: VERIFIED

### Criterion C4: a nearby confirmed track caps the reach
Test: test_a_nearby_confirmed_track_caps_the_reach[0.5-False]
Mutation: `_is_duplicate_of` uses the uncapped reach.
Result: VERIFIED

### Criterion C4b: with the cap off (inf), the same duplicate is merged
Test: test_a_nearby_confirmed_track_caps_the_reach[inf-True]
Mutation: the tracker ignores the cap it was given and always uses 0.5.
Result: VERIFIED. The first mutation tried (hardcode 0.5 in the `min` line) survived, because with cap inf `_capped` returns through its no-cap shortcut before reaching that line: the mutant was identical to the real code, so it said nothing about the test.

### Criterion C5: the reach grows with frames searched
Test: test_the_reach_grows_with_frames_searched[1.0-True]
Mutation: the growth term is dropped from `_reach`.
Result: VERIFIED

### Criterion C5b: the reach is the one the tracker was given
Test: test_the_reach_grows_with_frames_searched[0.0-False]
Mutation: `_reach` uses the `MERGE_REACH` constant instead of `merge_reach`.
Result: VERIFIED

### Criterion C6: a duplicate beyond the default reach keeps its own ID
Test: test_a_duplicate_beyond_the_default_reach_keeps_its_own_id
Mutation: default `MERGE_REACH = 20.0`.
Result: VERIFIED

### Criterion C7: merging is on by default
Test: test_duplicate_born_while_the_original_searches_ends_with_the_original_id
Mutation: default `MERGE_REACH = 0.0`.
Result: VERIFIED

### Criterion C8: the CLI passes --merge-reach and --merge-reach-growth to the tracker
Test: test_cli_passes_the_merge_reach_to_the_tracker
Mutation: the CLI parses the flags but doesn't pass them to `Tracker`.
Result: VERIFIED

## Verification results
12/12 criteria VERIFIED. (Run before the review's rename of `_duplicates` to `_is_duplicate_of` and the reach's move into `_reach`; both are pure refactors and the 12 tests still pass.)

Measurement criteria (the duplicate counts before building, ID kept and gap recovery before and after, the cap's bind rate, MOT17 per video at both input sizes) are benchmark results, not unit-testable; they are in [antiuav-duplicates.md](../benchmarks/antiuav-duplicates.md).
