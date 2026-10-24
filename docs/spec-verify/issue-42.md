# spec-verify: issue #42, scoring any detector on the leak-free Anti-UAV val

Spec: the acceptance criteria in [issue #42](https://github.com/Khriis-K/Tether/issues/42) and the scoring choices of [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md). Built as `tether.antiuav_split` (the split), `tether.antiuav_detect` (the detection dump) and `tether.drone_score` (the scorer).
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Tests are in `python/tests/test_antiuav_split.py`, `test_antiuav_detect.py` and `test_drone_score.py`. 25 of 26 mutations were caught (20 first, then 3 and 3 more for the two rounds of review fixes); the survivor is an equivalent mutant (C8).

### Criterion C1: the split holds out whole sessions, and none is in two splits
Test: test_a_session_is_the_clip_name_without_its_last_two_fields, test_no_session_is_in_two_splits
Mutation: a session is the whole clip name, so clips of one session are dealt out to both sides.
Result: VERIFIED
Mutation: val fills up to 20% instead of stopping once it reaches 15%.
Result: VERIFIED

### Criterion C2: test is identical to the official test
Test: test_test_is_exactly_the_official_test
Mutation: the last test clip is dropped.
Result: VERIFIED

### Criterion C3: val has sessions from both recording days
Test: test_val_has_sessions_from_both_recording_days (20 seeds)
Mutation: sessions are drawn day after day instead of alternating, so the first day fills val alone.
Result: VERIFIED

### Criterion C4: reproducible from its seed
Test: test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one; the committed `splits/antiuav.json` is re-derived from its own seed and frame counts in test_the_committed_split_is_reproduced_from_its_own_seed_and_frame_counts
Mutation: the generator is unseeded.
Result: VERIFIED

### Criterion C5: detections from any model can be dumped for every clip of a split, in the detection file format
Test: test_every_clip_of_the_split_gets_a_detection_file, test_the_detector_is_given_the_model_score_floor_output_and_clip_video, test_a_detection_file_with_the_wrong_number_of_frames_is_refused, test_clips_already_dumped_in_full_are_skipped
Mutation: the score floor isn't passed to the detector.
Result: VERIFIED
Mutation: no frame-count check after a clip is dumped.
Result: VERIFIED
Mutation: clips already dumped in full are run again.
Result: VERIFIED

### Criterion C6: null cases: nothing, and the same box every frame, score AP50 about 0; copying the labels scores 1
Test: test_a_detector_that_outputs_nothing_scores_zero, test_a_detector_that_outputs_the_same_box_every_frame_scores_about_zero, test_a_detector_that_copies_the_labeled_boxes_scores_one, test_a_detector_that_boxes_the_whole_frame_finds_nothing
Mutation: the IoU rule is dropped, so any detection on a drone frame finds the drone.
Result: VERIFIED (the fixed-box test fails)
Mutation: a detector with no detections scores 1.
Result: VERIFIED

### Criterion C7: a hit is IoU >= 0.5, and a second box on the drone is a false positive
Test: test_a_detection_is_a_hit_from_iou_of_one_half, test_a_second_box_on_the_drone_is_a_false_positive
Mutation: IoU strictly above 0.5.
Result: VERIFIED
Mutation: every detection that overlaps the drone counts toward it, none as a false positive.
Result: VERIFIED

### Criterion C8: AP50 is the area under the precision envelope, with tied scores cut together
Test: test_ap50_is_the_area_under_the_precision_envelope, test_ap50_does_not_depend_on_the_order_of_tied_scores
Mutation: precision ignored (AP becomes the recall reached).
Result: VERIFIED
Mutation: one cut per detection instead of per distinct score.
Result: EQUIVALENT. Counts at each cut are taken with `searchsorted`, so a repeated cut counts every tied detection anyway and adds no recall; the area is unchanged. Not a test gap.

### Criterion C9: recall, precision and F1 at a threshold, and the best-F1 threshold
Test: test_the_best_f1_threshold_is_the_score_cut_with_the_highest_f1, test_detections_below_the_threshold_are_not_counted
Mutation: the threshold with the best precision instead of the best F1.
Result: VERIFIED

### Criterion C10: recall split by COCO-style box size at source resolution
Test: test_recall_is_split_by_coco_style_box_size_at_source_resolution
Mutation: the tiny/small edge moved from 16 to 8 px.
Result: VERIFIED

### Criterion C11: false alarms per minute count only drone-absent frames, at each clip's own frame rate
Test: test_false_alarms_count_only_frames_where_the_drone_is_labeled_absent, test_false_alarms_per_minute_use_each_clips_own_frame_rate
Mutation: detections on every frame count as false alarms.
Result: VERIFIED
Mutation: a fixed 20 fps for every clip.
Result: VERIFIED

### Criterion C12: the results file names the model, split, threshold and IoU rule
Test: test_the_command_saves_results_naming_model_split_threshold_and_iou_rule, test_a_given_threshold_replaces_the_best_f1_one
Mutation: the model is named after the detection folder, ignoring `dump.json`.
Result: VERIFIED
Mutation: the threshold's rule isn't recorded.
Result: VERIFIED

### Review fixes (after `/code-review`)
Test: test_a_clip_cut_short_mid_line_by_a_stopped_run_is_dumped_again
Mutation: a detection file cut off mid-line is read as-is (the resume crashes on it).
Result: VERIFIED

Test: test_a_rerun_with_other_settings_into_the_same_folder_is_refused
Mutation: no check of the stored `dump.json`, so old files get labeled with the new score floor.
Result: VERIFIED

Test: test_a_threshold_below_the_dumps_score_floor_is_refused
Mutation: thresholds below the dump's floor are scored.
Result: VERIFIED

### Second review fixes (with the baseline write-up)
Test: test_a_best_f1_cut_at_the_lowest_saved_score_is_flagged
Mutation: the flag is never set.
Result: VERIFIED

Test (`tools/drone_breakdown.py`, the by-class and by-session tables): test_each_found_drone_is_credited_to_the_class_of_its_best_scored_hit, test_recall_is_split_by_recording_session_over_drone_frames_only
Mutation: the drone is credited to its first hit, not its best-scored one.
Result: VERIFIED
Mutation: drone-absent frames count toward a session's drones.
Result: VERIFIED
