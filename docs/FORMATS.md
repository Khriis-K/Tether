# File formats

The tracker's input (detections) and output (tracks) can be saved to files, so the tracker can run and be compared without video or a model. The Python tracker, the C++ tracker, MOT17 and Anti-UAV scoring, and the ByteTrack comparison all read and write these files. Pipeline runs also write a [timing log](#timing-log), which `tether-timing` summarizes for both pipelines.

All three are [JSON Lines](https://jsonlines.org/): UTF-8, `\n` line endings, one JSON object per line, one line per frame. Line `i` is frame `i`, counted from 0. A frame with nothing in it still gets a line, so frame numbers never drift.

Boxes are `[x1, y1, x2, y2]` in pixels, measured from the top-left of the image (`x2 > x1`, `y2 > y1`). Classes are the detector's integer class IDs (COCO for YOLO26).

## Detection file

```json
{"frame": 0, "detections": [{"box": [10.0, 20.0, 30.0, 40.0], "score": 0.91, "class": 0}]}
{"frame": 1, "detections": []}
```

| Field | Type | Meaning |
|---|---|---|
| `box` | 4 numbers | Detected box |
| `score` | number, 0–1 | Detector confidence. Saved unfiltered (down to about 0.1). Each tracker applies its own thresholds. |
| `class` | integer | Detector class ID |

## Track file

```json
{"frame": 0, "tracks": [{"id": 1, "box": [10.0, 20.0, 30.0, 40.0], "class": 0, "status": "tracked", "predicted_box": [10.0, 20.0, 30.0, 40.0], "frames_since_seen": 0, "path": [[20.0, 30.0]]}]}
{"frame": 1, "tracks": []}
```

Tracks in each frame are sorted by `id`.

| Field | Type | Meaning |
|---|---|---|
| `id` | integer, starts at 1 | Stable track ID. Never reused within a run. |
| `box` | 4 numbers | The box of the detection the track last matched. For a `"searching"` track that's its last seen box. Its guess for where the object is now is `predicted_box`. |
| `class` | integer | Class of the detections the track follows |
| `status` | `"tentative"`, `"tracked"` or `"searching"` | Lifecycle state. Removed tracks are left out. |
| `predicted_box` | 4 numbers | Where the tracker expects the object in the next frame |
| `frames_since_seen` | integer | Frames since the track last matched a detection (0 means it matched this frame) |
| `path` | list of `[x, y]` | Box centers from the last 30 matched frames, oldest first |

## MOT results file

`tether-mot17` converts each track file to the [MOT Challenge](https://motchallenge.net/) results format so TrackEval can score it. It's not JSON: one CSV row per track per frame, one file per video.

```
1,3,10.00,20.00,20.00,40.00,1,-1,-1,-1
```

`frame,id,left,top,width,height,conf,-1,-1,-1`, with frames counted from 1. Only `"tracked"` people (COCO class 0) are written: tentative tracks aren't confirmed yet, and a searching track's `box` is where it was last seen, not where it is. Tracks carry no score, so `conf` is always 1. Boxes are written as-is, without shifting to MOT's 1-based pixel coordinates, the same as ByteTrack, so both trackers are scored on equal terms.

## Timing log

Every pipeline run writes one, in Python (`tether-demo`, default `results/timing-<date>-<time>.jsonl`) and in C++. `tether-demo --dump` only saves detections and isn't timed. Each line holds when every step of that frame started and ended.

```json
{"frame": 0, "steps": {"capture": [0.0, 2.1], "prep": [2.1, 3.4], "detect": [3.4, 21.0], "track": [21.0, 21.3], "draw": [21.3, 23.0]}}
{"frame": 1, "steps": {"capture": [23.1, 25.0], "prep": [25.0, 26.2], "detect": [26.2, 43.9], "track": [43.9, 44.1], "draw": [44.1, 45.8]}}
```

Times are `[start, end]` in milliseconds from a monotonic clock (Python's `time.perf_counter`, C++'s `std::chrono::steady_clock`). The clock's zero point doesn't matter, since only differences are used, but every line in a log must use the same clock. All five steps appear on every line, in any order:

| Step | Covers |
|---|---|
| `capture` | Getting the next frame from the camera or file. On a webcam this includes waiting for the camera. |
| `prep` | Turning the frame into model input (letterbox, color order, scaling). |
| `detect` | Running the model and turning its output into detections. |
| `track` | One tracker update. |
| `draw` | Drawing the overlay and putting the frame on screen. In Python this includes `cv2.waitKey(1)`, which is where OpenCV actually paints the window, and on Windows it can take a few ms. |

Steps are stored as times, not durations, so the parallel pipeline can use the same format: there, one frame's steps don't follow each other directly, and the next frame's capture overlaps this frame's detect.

`tether-timing LOG` computes, as mean and worst 1% (P99):

- **Frame time:** from one frame's `draw` end to the next frame's `draw` end. This is what FPS is (1000 / mean frame time).
- **End-to-end delay:** from a frame's `capture` end to its `draw` end, i.e. from frame in hand to frame on screen. It leaves out what happens before the frame reaches the program: the camera's own delay, and time a frame sat in the camera driver's buffer. If the pipeline runs slower than the camera, the driver may hand over old buffered frames instantly, so the real delay grows while the measured one doesn't. Check that frame time keeps up with the camera before trusting a webcam delay number.
- **Each step's time:** its end minus its start.

P99 is the nearest-rank value: sort the values and take number ⌈0.99 × n⌉ (counting from 1). It is always a real measured value; with fewer than 100 values it's the largest. Every frame counts, including the slow first ones (model warm-up, opening the window), so on short runs P99 mostly reports warm-up. Use runs of a few hundred frames or more. The results are saved as JSON next to the log (`timing-….summary.json`) or wherever `--out` says.

## Matching rules

Both trackers must follow these rules exactly, or their IDs will drift apart.

- **Score threshold.** Before matching, drop detections scored below the tracker's `min_score` (default 0.25). Column order below is the file order of the detections that remain.
- **Rows and columns.** Tracks are the rows, in ID order. Detections are the columns, in file order.
- **Valid pairs.** A track and a detection can match only if they have the same `class` and their IoU is at least 0.3.
- **Assignment.** Pick the one-to-one pairing with the most valid pairs. Among those, pick the one with the lowest total `1 − IoU`. In practice: give invalid cells a cost of 1e6, which is more than any total of valid costs, run Hungarian, and then drop the invalid pairs it chose.
- **Ties.** When two different assignments have exactly the same cost, the solver decides. Python uses SciPy's `linear_sum_assignment` (Crouse's shortest augmenting path). The C++ port must use the same algorithm over the same row and column order, so it breaks ties the same way. Exact ties need bit-equal float costs, so they mostly happen in symmetric, scripted scenes.
- **New tracks.** Unmatched detections start new tracks in file order and take the next unused ID.

## Determinism

The same detection file always gives a byte-identical track file. Comparisons between Python and C++ check IDs and statuses exactly and boxes within a small tolerance, because floating-point math and number printing differ slightly between the two languages.
