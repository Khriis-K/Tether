# spec-verify: issue #45, the transfer and bird false-alarm test sets

Spec: the acceptance criteria in [issue #45](https://github.com/Khriis-K/Tether/issues/45), [ADR 0004](../adr/0004-drone-detector-on-anti-uav.md), and the orchestrator's empty-sky addendum (a stated, testable rule, fixed on val before any test run). Built as:
- `tether.antiuav_overlay`: the Anti-UAV overlay check.
- `tether.near_duplicates`: pHash and dedup.
- `tether.transfer_sets`: the set builders and the overlay check on real frames.
- `tether.image_detect`: the image dump.
- `tether.empty_sky`: the empty-sky rule.
- `tether.drone_score`: the image-folder mode, the bird rate and `--empty-sky`.

Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. The tests run were `python/tests/test_antiuav_overlay.py`, `test_near_duplicates.py`, `test_transfer_sets.py`, `test_drone_score_images.py`, `test_drone_score.py`, `test_empty_sky.py` and `test_image_detect.py`. The first rounds also ran four tests on the real datasets. Those were taken out of the suite before commit, because they had to skip without the data, and the repo's commit check rejects skipped tests. M5 and M10 were rerun without them.

25 of 27 mutations are caught by the committed tests. Three were caught only after their tests were made stronger (in C1, C3 and C4). Of the two survivors, one is effectively equivalent (C1). The other is the overlay threshold set to 0 (C1), which only real Seraphim images can catch. On the data, that mutant would drop the 68 Seraphim images that score exactly 0, against 0 dropped in `splits/seraphim-test.json` (counted from the scan's per-image scores, which aren't committed).

### Criterion C1: the overlay check flags real Anti-UAV frames and no Seraphim image checked by eye
Test: the synthetic-frame tests in test_antiuav_overlay.py. Real frames are checked by the `tether-transfer-sets overlay-check` run (681 of 681 flagged, `docs/benchmarks/drone-transfer-overlay-check.json`) and the Seraphim scan (0 of 8,349, including the 432 checked by eye, `splits/seraphim-test.json`).
- Mutation: no background subtraction. Result: VERIFIED (a frame-wide grid is flagged).
- Mutation: only the whole image is scored, not the letterboxed 16:9 strip. Result: VERIFIED.
- Mutation: threshold raised to 60. Result: VERIFIED.
- Mutation: threshold lowered to 0. Result: SURVIVED in the committed tests. Synthetic negatives score below 0. Only real images score between 0 and 14 (all Seraphim images score 0 or below), so only the data run catches this one. A test on the eye-checked Seraphim images caught it while the data tests were in the suite.
- Mutation: line contrast measured against the band above only. Result: first SURVIVED. Added test_edges_along_the_reticles_places_are_not_lines (a bright panel whose border lies on the brackets), then VERIFIED.
- Mutation: line contrast as the mean of both sides instead of the smaller side in one direction. Result: SURVIVED, effectively equivalent. On a step edge the two sides cancel in a mean too, so it also rejects edges and keeps lines. The first run of this mutation was a syntax error; it was rewritten and rerun.

### Criterion C2: dropped Seraphim images are listed with the count
Test: test_seraphim_images_with_the_overlay_are_dropped_and_listed_with_their_score
- Mutation: every image kept. Result: VERIFIED.

### Criterion C3: Drone-vs-Bird BT images deduplicated by perceptual hash; kept list and count committed
Test: test_near_duplicates.py, test_only_bird_images_are_taken_and_near_duplicates_dropped_in_name_order
- Mutation: a near-duplicate means distance 0. Result: VERIFIED.
- Mutation: compared against dropped images too. Result: VERIFIED.
- Mutation: every file taken, not just `BT`. Result: VERIFIED.
- Mutation: names sorted as strings, not by number. Result: VERIFIED. At first only the data test caught it. The fixture was then changed so the string order keeps a different image, and the unit test catches it too.

### Criterion C4: the image-folder mode gives AP50 and recall/precision/F1, with #42's null cases
Test: the null-case and matching tests in test_drone_score_images.py
- Mutation: no IoU rule. Result: VERIFIED (the same box on every image scores above 0).
- Mutation: one detection can claim several drones. Result: VERIFIED.
- Mutation: detections claim in file order, not score order. Result: VERIFIED.
- Mutation: a detection claims the first drone over IoU 0.5, not the one it overlaps most. Result: first SURVIVED, because the test's second detection could take the other drone either way. Rewrote test_a_detection_claims_the_drone_it_overlaps_most so the wrong choice loses a drone, then VERIFIED.
- Mutation: YOLO label width and height swapped. Result: VERIFIED.

### Criterion C5: the per-image bird false-alarm rate (none gives 0%, a box on every image 100%)
Test: test_no_detections_is_a_false_alarm_rate_of_zero, test_a_box_on_every_image_is_a_false_alarm_rate_of_one, test_an_image_counts_once_however_many_boxes_it_has, test_boxes_below_the_threshold_raise_no_alarm, test_the_command_refuses_a_threshold_below_the_dumps_floor
- Mutation: detections counted, not images. Result: VERIFIED.
- Mutation: a box exactly at the threshold doesn't count. Result: VERIFIED.
- Mutation: the score floor isn't checked for drone-free images. Result: VERIFIED.

### Criterion C6: the empty-sky rule is stated, testable, and listed for val and test
Test: test_empty_sky.py, test_with_empty_sky_frames_given_only_those_frames_count_as_false_alarms, test_an_empty_sky_frame_with_the_drone_labeled_is_refused
- Mutation: the frames outside the clip don't count as possibly holding the drone. Result: VERIFIED. The first form of this mutation, with edges at ±10⁹, made a list of billions of frames and hung; the run was killed, the file restored, and the mutation rerun with edges just past the margin.
- Mutation: the margin is off by one. Result: VERIFIED.
- Mutation: `--empty-sky` ignored. Result: VERIFIED.
- Mutation: empty-sky minutes taken from every drone-absent frame. Result: VERIFIED.
- Mutation: a listed frame with the drone labeled is accepted. Result: VERIFIED.

### Criterion C7: the COCO baseline is scored on both sets, from images run in list order
Test: test_image_detect.py
- Mutation: images linked in sorted, not list, order. Result: VERIFIED.
- Mutation: the dump's line count isn't checked. Result: VERIFIED.
- Mutation: a complete dump is run again. Result: VERIFIED.
