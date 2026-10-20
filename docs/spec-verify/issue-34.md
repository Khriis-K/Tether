# spec-verify: issue #34, the single-target prior

Spec: the acceptance criteria in [issue #34](https://github.com/Khriis-K/Tether/issues/34): an explicit option, off by default; with it on, if exactly one track is searching and none is tracked, that track matches the best (nearest) high detection anywhere in the frame, ignoring the IoU, gate, cap and size limits; it never fires with more than one confirmed track. Built as `Tracker(single_target=True)` / `tether-track --single-target`.
Each row: baseline run on the real code, then one targeted mutation, then the source file restored and byte-compared before the next. All mutations ran and were restored before any benchmark started. Tests are in `python/tests/test_single_target.py`; mutations are in `python/tether/tracker.py` unless noted.

### Criterion C1: off by default, so behavior with the option unset is unchanged
Test: test_off_by_default_a_far_return_starts_a_new_track. Also checked outside the suite: with defaults, 15 existing detection files (3 MOT17, 8 Anti-UAV val at 20% deletion, 4 with 1 s gaps) give byte-identical track files on `main` and on this branch.
Mutation: `single_target` defaults to True.
Result: VERIFIED

### Criterion C2: with it on, the only searching track takes a detection anywhere, of any size
Test: test_on_the_only_searching_track_takes_a_far_detection_of_any_size (30 box sizes away, 3× taller)
Mutation a: the rule keeps a gate (1.75 box sizes).
Result: VERIFIED
Mutation b: the rule uses the search gate's matcher (capped gate and size check).
Result: VERIFIED

### Criterion C3: it takes the nearest detection
Test: test_on_it_takes_the_nearest_detection (the farther one listed first)
Mutation: it takes the first detection, not the nearest.
Result: VERIFIED

### Criterion C4: it never fires when more than one track is searching
Test: test_on_it_never_fires_with_two_searching_tracks
Mutation: the one-confirmed-track check is dropped, so every searching track gets an unlimited match.
Result: VERIFIED

### Criterion C5: it never fires while any track is tracked
Test: test_on_it_never_fires_while_another_track_is_tracked
Mutation: only searching tracks are counted, so a tracked one doesn't block it.
Result: VERIFIED

### Criterion C6: `tether-track --single-target` turns it on, and it's off without the flag
Test: test_cli_flag_turns_it_on
Mutation (`python/tether/cli.py`): the flag is parsed but not passed to the tracker.
Result: VERIFIED
