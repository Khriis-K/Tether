# spec-verify: issue #11, the C++ sequential pipeline

Spec: the acceptance criteria in [issue #11](https://github.com/Khriis-K/Tether/issues/11): webcam and video file through one video-source interface, with file mode uncapped; CPU frame prep behind a frame-prep interface; the ONNX Runtime detector behind a detector interface, with model size as a setting; per-step timing logs in the shared format that work with the existing summary tool; dropped frames counted in webcam mode; the whole-pipeline Python vs. C++ comparison saved to a results file; detection dumps in the shared format. Plus the issue comment: frame prep takes its input height and width from the model, and non-square inputs work.

Each mutation was applied to one source file, the tests rebuilt and run (Release, MSVC, or pytest for Python), and the file restored before the next. A mutation counts as caught when at least one test fails. C++ tests are in `cpp/tests/`, Python tests in `python/tests/`.

## Results

31 mutations, 30 caught. One survived the first run and exposed a test gap, now fixed:

- **Negative drop counts** (dropping the clamp at 0) survived: no test had a gap under half a camera period. Added `FramesDropped.AGapShorterThanHalfAPeriodIsNoneNotNegative`.

One survives, and is left that way:

- **The tracker run outside the timed `track` span.** The loop tests use a fake clock that only moves when a fake step says so, and the real tracker can't advance it, so `track` is 0 ms either way. Catching it would need a test-only seam around the tracker, or a real-clock test with a timing threshold that could be flaky. The real runs are the check: their logs show `track` from 0.01 ms (Anti-UAV) to 13.9 ms (Python, MOT17-04 at 1280×736), in line with #10's tracking-step timings (`docs/benchmarks/cpp-pipeline-speed.md`).

| Criterion | Mutation | Caught by (example) |
|---|---|---|
| Frame prep matches Python | sizes round half away from zero (`std::round`) | `Letterbox.RoundsHalvesToEvenLikePython` |
| | padding not centered | `Letterbox.*`, `CpuFramePrep.*` |
| | red and blue swapped | `CpuFramePrep.BlobIsRgbChannelsFirstScaledToOne` |
| | scaled by 1/256 | same |
| Detector decoding | box math in double, not float32 | `Decode.DoesTheMathInFloat32AsNumpyDoes` |
| | score threshold exclusive | `Decode.KeepsScoresAtTheThresholdAndDropsOnesBelow` |
| | boxes left empty by clipping kept | `Decode.ClipsBoxesToTheFrameAndDropsOnesLeftEmpty` |
| | no clip at the frame's far edge | same |
| Input size from the model | dynamic (-1) dimensions accepted | `ModelInputSize.RejectsDynamicOrMalformedShapes`, `OnnxDetector.RejectsAModelWithADynamicInputSize` |
| | height and width swapped (non-square input) | `ModelInputSize.*`, `OnnxDetector.*`, `PipelineCli.DumpWritesEveryFramesDetections` |
| Dropped frames | gaps floored instead of rounded | `FramesDropped.TimingJitterRoundsToTheNearestWholeFrame` |
| | negative counts allowed | `FramesDropped.AGapShorterThanHalfAPeriodIsNoneNotNegative` (added) |
| | a file reports 0 drops instead of none | `VideoSource.AFileNeverReportsDroppedFrames`, `PipelineCli.TrackingWritesATimingLineForEveryFrame` |
| | the loop doesn't pass the count on | `RunSequential.EachFrameCarriesTheSourcesDroppedCount` |
| | a camera with no frame rate or timestamps reads as 0 drops | `DropCounter.UnknownWhenTheCameraReportsNoFrameRate`, `DropCounter.UnknownWhenTheCameraGivesNoTimestamps` (added after code review; written test-first) |
| | `"dropped"` written on every line | `TimingLogWriter.WritesOneLinePerFrameWithEveryStepInTheSharedFormat` |
| | `tether-timing` leaves out the total | `test_the_summary_of_a_webcam_run_includes_its_dropped_frames` |
| | a file run's total is 0, not absent | `test_a_file_run_has_no_dropped_count` |
| Pipeline loop | the stop from the draw step ignored | `RunSequential.StopsAfterTheFrameWhereShowSaysStop` |
| | `tracker.update` outside the timed span | **survives** (see above) |
| | the dump ignores the stop | `DumpDetections.StopsWhenAskedBeforeTheNextFrame` |
| Shared file formats | lines not flushed as written | `TimingLogWriter.EachLineIsOnDiskAsSoonAsItIsWritten`, `DetectionWriter.EachLineIsOnDiskAsSoonAsItIsWritten` |
| Settings | `--max-frames` off by one | `PipelineCli.MaxFramesStopsTheRunEarly` |
| | `--max-frames 0` accepted | `PipelineCli.BadArgumentsExitWithTwo` |
| | `--model` ignored | `PipelineCli.DumpWritesEveryFramesDetections` and 2 more |
| | any model size accepted | `ModelForSize.PicksTheYolo26ModelOfThatSize`, `PipelineCli.AnUnknownModelSizeIsABadArgument` |
| Python vs. C++ comparison | FPS ratio inverted | `test_speedups_are_python_time_over_cpp_time` |
| | no guard for a step C++ spends no time on | same (division by zero) |
| | `tether-demo --headless` ignored | `test_headless_mode_tracks_and_logs_every_frame_without_a_window` |
| | different frame counts not caught | `test_main_fails_when_the_pipelines_saw_different_frame_counts` |
| | the installed `tether` run instead of this checkout's | `cpp_pipeline_check.py::test_both_real_pipelines_run_on_the_same_frames` (ctest `pipeline_compare`) |
| | timing logs mixed up | `test_main_runs_both_pipelines_on_the_same_video_and_saves_the_comparison` |

## Checked by hand, not by tests

- **Webcam mode** (`tether-pipeline 0`): the laptop camera through MSMF, 300 frames. The drop count (69) matched an independent estimate from elapsed time (12,257 ms ÷ 33.33 ms ≈ 367.7 periods, about 368 frames captured, 300 read → 69 missing). Rechecked after code review moved the counting into `DropCounter`: 150 frames, 33 counted, 33 expected.
- **File mode is uncapped**: neither source sleeps. A 20 FPS clip ran at 25.1 FPS, faster than its own frame rate.
- **Same work as Python**: detection dumps from both pipelines are byte-identical on 1,525 real frames (`docs/benchmarks/cpp-pipeline-speed.md`).
