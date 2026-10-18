# spec-verify: issue #3, track lifecycle

Spec: the acceptance criteria in [issue #3](https://github.com/Khriis-K/Tether/issues/3) and SPEC.md user stories 4, 8, 24, 25, 26, "Tracker (core module)" and the "Seams" behavior tests.
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_lifecycle.py` unless noted.

### Criterion C1: statuses tentative, tracked and searching appear in the output; removed tracks drop out
Tests: test_new_detection_starts_as_a_tentative_track, test_confirmed_track_that_loses_its_detection_is_searching_and_counts_missed_frames, test_track_is_removed_once_missing_longer_than_the_limit
Mutation a: tracker.py, a confirmed track that misses still reports "tracked".
Result: VERIFIED
Mutation b: tracker.py, missed tracks are never kept (the old one-miss-and-gone rule).
Result: VERIFIED

### Criterion C2: confirm and remove thresholds are settings counted in frames
Tests: test_track_is_confirmed_after_the_confirm_threshold_in_frames (confirm 1, 2, 3, 5), test_track_is_removed_once_missing_longer_than_the_limit (limit 1, 5, 12), test_cli_passes_lifecycle_settings_to_the_tracker
Mutation a: tracker.py, confirm one frame late (`hits > confirm_frames`).
Result: VERIFIED
Mutation b: tracker.py, remove one frame early (`<` instead of `<=`).
Result: VERIFIED
Mutation c: tracker.py, remove one frame late (`<= limit + 1`).
Result: VERIFIED
Mutation d: cli.py, flags parsed but not passed to the tracker.
Result: VERIFIED

### Criterion C3: a one-frame false detection never becomes a confirmed track
Tests: test_one_frame_false_detection_never_becomes_a_confirmed_track, test_tentative_track_that_misses_a_frame_is_dropped_before_confirming
Mutation a: tracker.py, every new track is confirmed at once.
Result: VERIFIED
Mutation b: tracker.py, tentative tracks survive misses like confirmed ones.
Result: VERIFIED

### Criterion C4: an object that disappears for N frames (fewer than the limit) and comes back keeps its ID
Tests: test_object_that_disappears_for_fewer_frames_than_the_limit_keeps_its_id (gaps 1, 5, 12 with limit 12), test_stationary_object_that_disappears_and_returns_keeps_its_id, test_searching_track_keeps_predicting_along_its_motion_and_keeps_its_last_seen_box
Mutation a: tracker.py, searching tracks stop predicting (the filter freezes at the last seen frame).
Result: VERIFIED
Mutation b: tracker.py, `frames_since_seen` not reset when a searching track is matched again.
Result: VERIFIED

### Criterion C5: a track gets removed after the disappearance limit
Tests: test_track_is_removed_once_missing_longer_than_the_limit, test_default_settings_confirm_and_remove_tracks
Covered by C2's mutations b and c.
Result: VERIFIED

### Criterion C6: two objects crossing paths don't swap IDs
Tests: test_two_objects_crossing_while_one_hides_the_other_do_not_swap_ids (new: the back object is undetected for 7 frames mid-crossing), plus test_motion.py::test_fast_objects_crossing_paths_do_not_swap_ids from issue #2.
Mutation: tracker.py, searching tracks are matched on their last seen box, not their prediction.
Result: VERIFIED

### Criterion C7: output stays deterministic
Test: test_cli.py::test_same_input_run_twice_gives_byte_identical_output (now runs through the full lifecycle). Its mutation was verified in issue #2; nothing in this change adds a source of nondeterminism (no randomness, no hash-order iteration).
Result: VERIFIED (by issue #2's mutation)
