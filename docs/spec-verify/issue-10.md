# spec-verify: issue #10, the Python-vs-C++ match check and tracking-step timing

Spec: the acceptance criteria in [issue #10](https://github.com/Khriis-K/Tether/issues/10): run both trackers on the same input files; pass only when IDs and statuses match exactly on every frame and boxes are within the tolerance; on a mismatch, report the first differing frame and track; run on the scripted test inputs and on at least one real detection file; time the tracking step alone in both languages and save the speedup to a results file.

Each mutation was applied to one source file (`python/tether/parity.py`, or `cpp/src/tether/timing.cpp` with the C++ tests rebuilt, Release, MSVC), the tests run, and the file restored before the next. A mutation counts as caught when at least one test fails. Python tests are in `python/tests/test_parity.py`, C++ tests in `cpp/tests/test_timing.cpp`.

## Results

31 mutations (22 Python, 9 C++); 30 caught, one accepted (below). Four survived the first runs and exposed test gaps, now fixed:

- **The Python timer not timing the tracker** (the `update` call removed from the timed loop) survived: "every time is above 0" holds even for an empty loop. A test now swaps in an `update` that sleeps 2 ms and checks 20 frames take at least 40 ms.
- **Min instead of median of the repeats** survived because the stand-in timings were all equal. A test now gives unequal timings and checks the median is kept.
- **The C++ clock started after the loop** survived for the same reason: two clock reads a few nanoseconds apart are still above 0. The test now requires over 0.05 ms for 5,000 frames (10 ns per tracker update, far below any real one).
- **`--repeats 2x` accepted as 2** survived; the bad-argument test now includes it.

Code review then found that a NaN box would pass (`abs(nan - x) > tolerance` is False). The check now uses `<=`, which fails on NaN, and two tests pin it. A failing C++ binary now stops with its error message, also tested.

| Criterion | Mutation | Caught by (example) |
|---|---|---|
| IDs match exactly | IDs not compared | `test_different_id_is_reported_at_its_frame_and_track` |
| Statuses match exactly | statuses not compared | `test_different_status_is_reported_at_its_frame_and_track` |
| (stricter than asked) | class / missed-frame count / path length not compared | `test_class_missed_frames_and_path_length_must_match_exactly` |
| Boxes within the tolerance | tolerance 1e6× looser | `test_a_box_beyond_the_tolerance_is_reported` |
| | predicted box not compared | same, `[predicted_box]` |
| | path points not compared | `test_a_path_point_beyond_the_tolerance_is_reported` |
| | `> tolerance` (NaN passes) | `test_a_nan_box_is_a_mismatch` (added) |
| | a track only Python has not reported | `test_a_track_only_one_side_has_is_reported` |
| | frame count not compared | `test_a_different_frame_count_is_reported_where_one_side_ends` |
| First differing frame and track | last mismatch reported instead of first | `test_only_the_first_differing_frame_is_reported` |
| | exit 0 on a mismatch | `test_main_fails_and_points_at_the_first_differing_frame` |
| | `all_match` always true | same |
| | C++ failure ignored | `test_a_failing_cpp_binary_stops_with_its_error` (added) |
| Scripted inputs | `--scripted` ignored | `test_main_passes_and_saves_the_speedup_when_outputs_match` |
| Real files | (the ctest `parity_scripted` and the runs in `docs/benchmarks/cpp-tracking-speed.md` use the real binaries) | |
| Time the tracking step alone | Python loop doesn't call `update` | `test_time_python_times_the_tracker_updates` (added) |
| | C++ clock started after the loop | `TimeTracking.ALongerInputTakesTime` (tightened) |
| | one repeat too few | `TimeTracking.GivesOneTimePerRepeat` |
| | median replaced by min | `test_main_keeps_the_median_of_the_repeats` (added) |
| Save the speedup | per-file speedup inverted | `test_main_passes_and_saves_the_speedup_when_outputs_match` |
| | overall speedup inverted | same |
| | `max_box_difference` ignores predicted boxes | `test_max_box_difference_covers_boxes_and_predicted_boxes` |
| C++ timing CLI | zero repeats allowed | `TimeCli.RejectsBadArguments` |
| | default repeats 1 | `TimeCli.RepeatsDefaultToFive` |
| | `--repeats=N` not split | `TimeCli.AcceptsRepeatsWithEquals` |
| | trailing junk accepted | `TimeCli.RejectsBadArguments` (tightened) |
| | two files accepted | same |
| | missing file exits 0 | `TimeCli.MissingFileIsAnError` |
| | frame count off by one | `TimeCli.PrintsFrameCountAndOneTimePerRepeat` |

## Accepted survivor

- **One scripted scene removed** (`far-jump`): the tests check that the scenes as a whole reach every status and several IDs, not a fixed list, and the other scenes still do. Pinning the exact list would only test the list.

## The end-to-end check

`ctest` runs `parity_scripted`: `tether-parity --scripted` against the binaries just built. Making the C++ tracker confirm a track one frame later made it fail with `steady: ... MISMATCH at frame 2, track 1: status tracked (Python) vs tentative (C++)`; restoring it made it pass.
