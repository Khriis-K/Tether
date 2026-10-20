# spec-verify: issue #31, tentative tracks surviving a miss

Spec: the acceptance criteria in [issue #31](https://github.com/Khriis-K/Tether/issues/31): a tentative track survives one miss, a one-off false detection still never gets confirmed, and the confirmation threshold is respected (on net hits, matched minus missed frames). Plus two follow-ons in the duplicate merge: a merged tentative is dropped, and a track only takes over a younger one (the age caveat in [antiuav-duplicates.md](../benchmarks/antiuav-duplicates.md)).
Each row: baseline run on the real code, then one targeted mutation, then `python/tether/tracker.py` restored and byte-compared before the next (scratch harness `results/i31/mutate.py`). Tests are in `python/tests/test_tentative_miss.py` unless noted.

### Criterion C1: a tentative track survives one miss and keeps its ID
Test: test_tentative_track_survives_one_missed_frame_and_keeps_its_id
Mutation: a tentative track's survival limit is 0 (the old rule).
Result: VERIFIED

### Criterion C1b: it is dropped on its second miss in a row
Test: test_tentative_track_is_dropped_on_its_second_missed_frame_in_a_row
Mutation: the limit is `tentative_misses + 1`.
Result: VERIFIED

### Criterion C2: a one-off false detection is never confirmed, and is gone two frames later
Test: test_one_off_false_detection_is_never_confirmed
Mutation: tentative tracks never die (limit `math.inf`).
Result: VERIFIED

### Criterion C2b: a detection flickering every other frame is never confirmed
Test: test_detection_that_flickers_every_other_frame_is_never_confirmed
Mutation: confirmed on total hits instead of net hits.
Result: VERIFIED

### Criterion C3: the confirmation threshold is respected on net hits
Test: test_a_missed_frame_takes_back_one_hit_before_confirming (confirm_frames 2, 3, 5)
Mutation 1: confirmed on total hits instead of net hits.
Mutation 2: `net_hits >= confirm_frames` → `net_hits > confirm_frames`.
Result: VERIFIED (both)

### Criterion C4: `tentative_misses` 0 restores the streak rule
Test: test_zero_tentative_misses_drops_a_tentative_track_on_its_first_miss
Mutation: the setting is ignored (always 1).
Result: VERIFIED

### Criterion C5: a tentative track is never merged into a younger track
Test: test_a_tentative_track_is_never_merged_into_a_younger_track
Mutation: the `old.id < new.id` check is dropped from `_is_duplicate_of`.
Result: VERIFIED

### Criterion C6: a merged tentative is dropped, not left as a ghost
Test: test_merge.py::test_duplicate_born_while_the_original_searches_ends_with_the_original_id
Mutation: merged-away tentatives are counted as missing a frame (so they survive it).
Result: VERIFIED

### Criterion C7: a duplicate that missed a frame is merged only once its net hits confirm it
Test: test_a_duplicate_that_missed_a_frame_is_merged_only_once_its_net_hits_confirm_it
Mutation: the merge fires on `hits + 1` instead of `net_hits + 1`.
Result: VERIFIED. This was VACUOUS against the first test set (no test had a duplicate miss a frame before confirming); the test was added and the mutation re-run.

## Verification results
9/9 criteria VERIFIED (10 mutations).
