# spec-verify: issue #9, the C++ tracker port

Spec: the acceptance criteria in [issue #9](https://github.com/Khriis-K/Tether/issues/9): a CMake build on Windows with no Windows-only code in the tracker; a C++ CLI that reads a detection file and writes a track file in the shared formats; Kalman prediction, Hungarian matching on predicted boxes and the frame-based lifecycle; every Python scripted behavior test ported and passing; deterministic output.

Each mutation was applied to one C++ source file, the tests rebuilt and run (Release, MSVC), and the file restored before the next. A mutation counts as caught when at least one test fails. Tests are in `cpp/tests/`; sources are in `cpp/src/tether/`.

## Results

39 mutations, all caught. Two survived the first run and exposed test gaps, now fixed:

- **A merge one frame early** (a tentative track merged when its next hit would *not* yet confirm it) survived: no test pinned when a merge happens. The Python suite has the same gap. Added `Merge.NothingIsMergedBeforeTheDuplicateWouldBeConfirmed`.
- **The CLI accepting trailing junk in a number** (`--min-score 0.5x` read as 0.5) survived because the bad-argument test pointed at files that didn't exist, so the CLI failed for that reason instead. The test now uses real files and expects exit code 2.

| Criterion | Mutation | Caught by (example) |
|---|---|---|
| Kalman prediction | position doesn't move by velocity | 14 tests, e.g. `Motion.*`, `CoastDamping.*` |
| | the measurement is ignored on update | 20 tests |
| Hungarian on predicted boxes | invalid pairs the solver chose are kept | 25 tests |
| Same ties as SciPy | no preference for a free column on a tie | `Assignment.MatchesScipyIncludingTiesAndRectangularShapes` |
| | columns scanned forward instead of in reverse | same |
| | transposed result not re-sorted by row | same |
| Matching cascade | tentative tracks match in stage 1 | `SearchGate.SearchingTrackTakesItsDetectionBackFromTheTentativeTrackItStarted` |
| | the one-hit distance stage is off | `Motion.ObjectAlreadyFastOnItsFirstFrameKeepsTheSameId` |
| Low-score pass | low pass uses MIN_IOU instead of LOW_MIN_IOU | `LowScore.LowDetectionBelowTheIouGateDoesNotKeepATrackTracked` |
| | low pass off | `LowScore.LowDetectionKeepsATrackedTrackTracked` |
| | low detections start tracks | 6 tests |
| | LOW_SCORE floor inclusive | `LowScore.DetectionAtOrBelowTheLowFloorIsIgnored` |
| Search gate | stage off | 14 tests |
| | "gate > 0" check off | `SearchGate.ZeroGateNeverMatchesEvenADetectionCenteredOnThePrediction` |
| | growth ignored | 9 tests |
| | size check always passes | `SearchGateSize.DetectionMuchBiggerOrSmallerStartsANewTrack` |
| | size compared to the coasted prediction | `SearchGateSize.SizeIsComparedToTheLastObservedBoxNotTheCoastedPrediction` |
| Nearest-track cap | cap off | 4 tests, e.g. `Cli.PassesTheCapToTheTracker`, `Merge.ANearbyConfirmedTrackCapsTheReach` |
| | cap measures to the track itself | 16 tests |
| | tentative tracks count toward the cap | 7 tests |
| Duplicate merge | merge off | 6 tests |
| | merged track keeps its own coasted motion | `Merge.MergedTrackKeepsFollowingTheObject` |
| | merges one frame early | `Merge.NothingIsMergedBeforeTheDuplicateWouldBeConfirmed` (added) |
| | reach growth ignored | `Merge.TheReachGrowsWithFramesSearched` |
| Lifecycle | confirm one frame late | 9 tests |
| | a tentative track survives a miss | 14 tests |
| | removal one frame early | `Lifecycle.ObjectThatDisappearsForFewerFramesThanTheLimitKeepsItsId` |
| | path not capped at 30 | `Tracker.RecentPathIsCapped` |
| Coast damping | damping ignored | `CoastDamping.VelocityShrinksByTheDampingEachCoastedFrame` |
| | matched frames damped too | `CoastDamping.FirstCoastedStepIsDampedToo` |
| Shared formats | numbers always in fixed notation | `Formats.NumbersArePrintedTheWayPythonPrintsThem` |
| | CRLF line endings | `Formats.FilesUseLfLineEndingsOnEveryPlatform` |
| | frame order not checked | `Formats.FrameNumbersOutOfOrderAreAnError` |
| CLI mirrors Python | `--coast-damping`, `--search-gate-cap`, `--merge-reach`, `--min-score`, `--max-missing-frames` each parsed but not passed on (5 mutations) | the matching `Cli.Passes*` test |
| | trailing junk in a number accepted | `Cli.BadArgumentsAreAnErrorWithAMessage` (tightened) |

## Outside the mutation run

- **Portable build:** the same sources and tests compile with GCC 13.3 (`-Wall -Wextra -Wpedantic`, no warnings after three fixes) under WSL Ubuntu and all tests pass there; Linux and Windows output on MOT17-09 is byte-identical. The CMake script itself was only run on Windows: WSL had no CMake installed.
- **Matches Python on real data:** see `docs/benchmarks/cpp-port-parity.md`.
- **Determinism:** `Cli.SameInputRunTwiceGivesByteIdenticalOutput` and `Tracker.OutputIsDeterministic`.
- **Debug build** (MSVC checked iterators) passes too.
