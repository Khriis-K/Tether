# spec-verify: issue #44, end-to-end drone tracking with real detections

Spec: the acceptance criteria in [issue #44](https://github.com/Khriis-K/Tether/issues/44) and the tracking choices of [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md). Built as `evaluate_detections` and `--detections` in `tether.antiuav`, and the low-score band rule (`low_score_ratio`) in both trackers and both `tether-track` CLIs.
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. Python tests are in `python/tests/test_antiuav_detections.py` and `test_low_score_band.py`; C++ tests in `cpp/tests/test_low_score_band.cpp`. 12 of 12 mutations were caught.

### Criterion C1: the evaluator scores the tracker on a folder of detection files with the ID-kept rule; the labeled-box mode is unchanged
Test: test_a_folder_that_copies_the_labels_scores_the_same_as_the_labeled_boxes, test_the_labeled_box_row_is_scored_on_the_same_clips_whatever_the_detector_did, test_missed_frames_count_against_the_id, test_classes_are_ignored_so_a_drone_seen_as_two_classes_keeps_one_id, test_a_detection_file_of_the_wrong_length_is_refused, test_the_tool_takes_the_clips_from_the_split_the_folder_was_dumped_for, test_the_tool_refuses_a_split_other_than_the_dumps; the labeled-box mode by the unchanged `test_antiuav.py`
Mutation: the detector row is scored only on frames where it gave a detection (the dropout rule), so its misses go unscored.
Result: VERIFIED (a detector that finds nothing would score n/a, not 0)
Mutation: the labeled-box row runs on the detector's detections.
Result: VERIFIED
Mutation: classes are kept, so a drone seen as two classes splits its ID.
Result: VERIFIED
Mutation: no frame-count check on a detection file.
Result: VERIFIED
Mutation: the split defaults to test instead of the one the folder was dumped for.
Result: VERIFIED
Mutation: a `--split` other than the dump's is accepted.
Result: VERIFIED
Added after code review: test_an_empty_split_is_refused_rather_than_scored_0_of_0 was written first and failed against the code without the guard (it scored 0 of 0 with no error), then passed with it.

### Criterion C2: a detection folder that copies the labeled boxes gives the same result as the labeled-box mode
Test: test_a_folder_that_copies_the_labels_scores_the_same_as_the_labeled_boxes
Covered by the C1 mutations: scoring only detected frames and running the labeled row on the detector's boxes both leave this test passing on copied labels (they agree there by construction), and are caught by the null-detector test beside it. So this test checks the equality the criterion asks for, and the null-detector test checks the equality isn't vacuous.
Result: VERIFIED (by the C1 rows)

### Criterion C3: the low-score band follows the cutoff by a stated rule, settable from the command line, tested at the default and at a moved cutoff
Test: test_at_the_default_cutoff_the_floor_is_bytetracks_0_1_exactly, test_at_the_default_cutoff_the_band_is_0_1_to_0_25, test_a_higher_cutoff_moves_the_floor_up_in_proportion, test_a_cutoff_under_0_1_still_has_a_band, test_the_ratio_can_be_set, test_a_ratio_of_one_turns_the_low_pass_off, test_the_cli_moves_the_band_with_min_score, test_the_cli_sets_the_ratio; the C++ `LowScoreBand.*` ports
Mutation (Python): the floor stays at 0.1 whatever the cutoff (the old rule).
Result: VERIFIED
Mutation (Python): the floor itself counts as low (`<=` for `<`).
Result: VERIFIED
Mutation (Python): the floor is a ratio of the default cutoff, not the one in use.
Result: VERIFIED
Mutation (Python): `--low-score-ratio` isn't passed to the tracker.
Result: VERIFIED
Mutation (C++): the floor stays at 0.1 whatever the cutoff.
Result: VERIFIED (4 of the 7 `LowScoreBand` tests fail)
Mutation (C++): `--low-score-ratio` is parsed but not stored.
Result: VERIFIED

### Criterion C4: results report ID kept with the real detector next to the labeled-box ceiling on the same clips
Test: test_results_name_the_model_and_its_score_floor_and_are_saved, test_the_labeled_box_row_is_scored_on_the_same_clips_whatever_the_detector_did
Covered by the "labeled-box row runs on the detector's detections" mutation above. Run on val: `docs/benchmarks/drone-tracking.md`.
Result: VERIFIED

### Criterion C5: the COCO baseline is tracked end to end on the new val and the result is saved
Not a code behaviour: the run is `docs/benchmarks/drone-tracking-coco-at-0.25.json` and `-coco-at-0.066.json` (10.9% and 13.2% ID kept), written up in `docs/benchmarks/drone-tracking.md`.
Result: DONE (no mutation applies)
