# spec-verify: issue #12, the HUD overlay

Spec: the acceptance criteria in [issue #12](https://github.com/Khriis-K/Tether/issues/12), plus SPEC.md user stories 2, 6–10, 16 and 17, and "Visual design":
- corner-bracket boxes and ID labels on tracked objects;
- per-track colours spread evenly around the colour wheel;
- fading trails from each track's recent path;
- dashed predicted boxes;
- searching tracks get an accent-coloured dashed box and a missing-for counter;
- status readable from line style alone (solid, dashed, accent dashed);
- pause and record-to-file both work.

Each mutation was applied to one source file. The tests were rebuilt (Release, MSVC) and the HUD, view, CLI and video-source suites run, then the file was restored before the next. A mutation counts as caught when at least one test fails, or the run hangs past 3 minutes. The tests are in `cpp/tests/test_hud.cpp`, `test_hud_view.cpp`, `test_pipeline_cli_hud.cpp` and `test_video_source_fps.cpp`.

## Results

31 mutations: 30 caught, and 1 survivor that exposed dead code, which was then removed.

- **The folder check in `open_recording` (removed).** Writing a recording to a folder failed with or without it, because OpenCV's writer won't open a folder either. The check was dead code, so it's gone. The test that a bad path gives exit 1 with "can't record" still passes without it.
- **Closing the window while paused** is caught by a hang, not a failure: the pause loop never ends. That's the real-world symptom too, a frozen app, and ctest would time it out.

| Criterion | Mutation | Caught by (example) |
|---|---|---|
| Colours spread evenly | hues step by 1/2, not the golden ratio | `TrackColor.AnyNumberOfTracksIsSpreadEvenly` |
| | the band around the accent not skipped | `TrackColor.NoTrackLooksLikeTheSearchingAccent` |
| | colour returned as RGB, not BGR | `TrackColor.IsTheHueAsABrightBgrColour` |
| Corner brackets and ID labels | brackets drawn as a full box | `DrawHud.ATrackedObjectGetsCornerBracketsNotAFullBox` |
| | brackets dashed | `DrawHud.ATrackedObjectsBracketsAreSolid` |
| | label not moved inside the box at the frame's top | `DrawHud.TheLabelStaysOnScreenForABoxAtTheTop` |
| | tentative tracks drawn | `DrawHud.TentativeTracksAreNotDrawn`, `PipelineCliRecord.SavesEveryFrameWithTheHudDrawn` |
| Dashed predicted boxes | predicted box solid | `DrawHud.ATrackedObjectsPredictedBoxIsDashedInItsColour` |
| | predicted box not drawn | same |
| Searching: accent dashed box and counter | drawn in the track colour | `DrawHud.ASearchingObjectIsAnAccentDashedBoxAtItsPredictionWithTheFramesMissing` |
| | drawn at the last seen box, not the prediction | same, `DrawHud.SearchingIsHeavierThanAPrediction` |
| | missing count left out of the label | same |
| Status from line style alone | searching as thin as a prediction | `DrawHud.SearchingIsHeavierThanAPrediction` |
| Fading trails | trail not faded | `DrawHud.TheTrailFadesFromTheNewestPointBackToTheOldest` |
| | trail fades the wrong way | same |
| Pause | PAUSED never shown | `DrawStatus.PausedAndRecordingEachShow`, `HudView.SpacePausesOnTheFrameUntilSpaceAgain` |
| | the paused mark left on after resuming | `HudView.SpacePausesOnTheFrameUntilSpaceAgain` |
| | quit ignored while paused | `HudView.QuittingWhilePausedStops` |
| | closing the window ignored while paused | `HudView.ClosingTheWindowWhilePausedStops` (hangs) |
| | Esc doesn't quit | `HudView.QOrEscStops` |
| | closing the window ignored | `HudView.ClosingTheWindowStops` |
| Record to file | REC indicator without the accent dot | `DrawStatus.PausedAndRecordingEachShow` |
| | nothing recorded while a window is shown | `HudView.RecordsEachDrawnFrameOnceWithoutTheStatusMarks` |
| | no window stops the run after one frame | `HudView.WithoutAScreenItOnlyRecordsAndNeverStops`, `PipelineCliRecord.*` |
| | the HUD view not used at all | `PipelineCliRecord.*` |
| | the source's frame rate ignored | `PipelineCliRecord.KeepsTheSourcesFrameRate` |
| | the fallback rate 25, not 30 | `PipelineCliRecord.AnImageSequenceRecordsAtThirtyFps` |
| | an image sequence's made-up 25 FPS trusted | `VideoSourceFps.AnImageSequenceCantTell` |
| | `--record` accepted with `--dump` | `PipelineCliRecord.ShowAndRecordDontGoWithDump` |
| | folder check removed | **survived**, and the check was removed (see above) |

## Checked by running the app

- **Recording frame rate:** the first real recording read back at 25 FPS, but MOT17-09 is 30 FPS (`seqinfo.ini`). OpenCV, through FFmpeg, reported 25 for the JPEG folder itself, while the test's PNG sequence reported nothing. Image sequences now report no rate (`VideoSourceFps.AnImageSequenceCantTell`), and the same run reads back at 30.
- **Recording:** `tether-pipeline` with `--record` on 300 frames of MOT17-09 (YOLO26n at 640) gave a 30 FPS, 300-frame MP4. Frame 250 is `docs/media/hud-mot17-09.jpg`. It shows brackets and labels on tracked people, thin dashed predictions, orange dashed searching boxes with "N  missing K", and trails.
- **Window:** `--show` ran on the same video, and the window kept drawing (the timing log kept growing). Space, q and Esc in the live window could **not** be pressed from this session, because the desktop was locked: Windows refused both the screenshot and the keystrokes. The key handling is covered by the scripted-screen tests above. The highgui wrapper in between (`WindowScreen`, about 15 lines) hasn't been run with a real key press.
