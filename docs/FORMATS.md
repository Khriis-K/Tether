# File formats

The tracker's input (detections) and output (tracks) can be saved to files, so the tracker can run and be compared without video or a model. The Python tracker, the C++ tracker, MOT17 and Anti-UAV scoring, and the ByteTrack comparison all read and write these files.

Both are [JSON Lines](https://jsonlines.org/): UTF-8, `\n` line endings, one JSON object per line, one line per frame. Line `i` is frame `i`, counted from 0. A frame with nothing in it still gets a line, so frame numbers never drift.

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
| `box` | 4 numbers | The track's current box |
| `class` | integer | Class of the detections the track follows |
| `status` | `"tentative"`, `"tracked"` or `"searching"` | Lifecycle state. Removed tracks are left out. |
| `predicted_box` | 4 numbers | Where the tracker expects the object in the next frame |
| `frames_since_seen` | integer | Frames since the track last matched a detection (0 means it matched this frame) |
| `path` | list of `[x, y]` | Recent box centers for trails, oldest first, capped in length |

## Determinism

The same detection file always gives a byte-identical track file. Comparisons between Python and C++ check IDs and statuses exactly and boxes within a small tolerance, because floating-point math and number printing differ slightly between the two languages.
