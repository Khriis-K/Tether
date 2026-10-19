# spec-verify: issue #8, Anti-UAV dropout simulator and scoring

Spec: the acceptance criteria in [issue #8](https://github.com/Khriis-K/Tether/issues/8) and SPEC.md user stories 51–54 and "Anti-UAV dropout tests".
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_antiuav.py`; every mutation is in `python/tether/antiuav.py`.

### Criterion C1: deletion rate and gap length are settings; gaps are given in seconds and converted with the clip's frame rate
Tests: test_gap_seconds_become_frames_at_the_clips_frame_rate, test_gaps_are_blanked_using_each_clips_own_frame_rate (a 10 fps clip), test_deletion_rates_zero_and_one_keep_all_and_none
Mutation a: seconds converted at a fixed 20 fps.
Result: VERIFIED
Mutation b: `evaluate` passes 20 fps instead of the clip's.
Result: VERIFIED
Mutation c: the deletion rate is ignored (always 0.2).
Result: VERIFIED

### Criterion C2: random deletion is seeded so runs can be repeated
Tests: test_random_deletion_is_repeatable_with_the_same_seed_and_differs_with_another, test_runs_with_the_same_seed_give_the_same_results, test_separate_runs_of_the_tool_delete_the_same_boxes (two processes, different PYTHONHASHSEED)
Mutation a: the generator is created unseeded.
Result: VERIFIED
Mutation b: each clip is seeded with Python's `hash()` of its name, which changes between processes. Survived the in-process test; the cross-process test was added to catch it.
Result: VERIFIED

### Criterion C3: "ID kept" requires IoU ≥ 0.5 with the labeled box
Tests: test_the_id_counts_only_where_the_track_overlaps_the_drone_with_iou_of_at_least_half, test_an_iou_of_exactly_half_holds_the_drone, test_a_track_that_covers_the_whole_image_never_keeps_the_id, test_tentative_and_searching_tracks_do_not_hold_the_id
Mutation a: threshold 0.3.
Result: VERIFIED
Mutation b: `>` instead of `>=`.
Result: VERIFIED
Mutation c: any status holds the drone (a searching track's box is stale).
Result: VERIFIED

### Criterion C4: reports the ID-kept rate under random deletion and the recovery rate across gaps
Tests: test_after_an_id_switch_only_the_most_common_id_counts, test_frames_whose_detection_was_deleted_are_not_scored, test_a_new_id_after_the_gap_is_not_recovered, test_recovery_is_checked_on_the_first_frame_the_drone_is_labeled_after_the_gap, test_a_gap_the_tracker_was_not_holding_before_is_counted_but_not_held, test_a_tracker_with_no_tracks_scores_zero_and_has_no_recovery_rate
Mutation a: every holding ID counts as kept, not just the most common one.
Result: VERIFIED
Mutation b: deleted frames are scored (which caps 20% deletion at 80%).
Result: VERIFIED
Mutation c: recovery checked on the frame after the gap even when the drone isn't labeled there.
Result: VERIFIED
Mutation d: any ID on the drone after the gap counts as recovered.
Result: VERIFIED
Mutation e: recovery divided by all gaps rather than held ones (an empty tracker would score 0% instead of "no rate").
Result: VERIFIED
Mutation f: gaps the tracker wasn't holding are scored anyway.
Result: VERIFIED

### Criterion C5: reports prediction error during gaps
Tests: test_a_perfect_prediction_has_zero_error_on_every_gap_frame, test_prediction_error_is_the_distance_from_the_previous_frames_prediction_to_the_drone, test_no_prediction_error_is_measured_for_a_gap_that_was_not_held
Mutation a: uses this frame's prediction (which is for the next frame).
Result: VERIFIED
Mutation b: error halved (not in pixels).
Result: VERIFIED

### Criterion C6: a degenerate tracker that freezes its box never scores as recovered
Test: test_a_tracker_that_freezes_its_box_is_never_recovered (exact on the drone up to the gap, then "tracked" with ID 1 forever, frozen)
Mutation: recovery checks only the ID, not the IoU.
Result: VERIFIED

### Criterion C7: results are saved to a file
Test: test_evaluates_the_split_and_saves_the_results
Mutation: the results file is not written.
Result: VERIFIED

### Supporting behavior
Mutation: labels read by `gt_rect` instead of `exist` (the real labels leave boxes on some absent frames). Killed by test_frames_marked_not_existing_have_no_drone_whatever_their_rect_says.
Mutation: no check that labels and video have the same length. Killed by test_a_clip_whose_labels_and_video_disagree_on_length_is_refused.
Mutation: a gap may end on the clip's last frame, leaving nothing to recover on. Killed by test_gaps_follow_settle_stretches_and_each_leaves_frames_after_it.
Result: VERIFIED
