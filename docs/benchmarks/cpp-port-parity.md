# C++ port parity with the Python tracker (#9)

**Question:** does the C++ tracker (`cpp/`, `tether-track`) give the same tracks as the Python one on real detection files, not just on the scripted tests?

**Answer:** yes, on everything we ran. On 87 files (the 7 MOT17 train videos and 80 Anti-UAV files: 40 test-split clips under two dropout conditions), every frame has the same track IDs, statuses, classes, `frames_since_seen` and path lengths in both, and the largest difference in any box or predicted box is 2.6e-9 px.

## Setup

- Both trackers at their defaults, run through their CLIs (`python -m tether.cli` and `cpp/build/Release/tether-track`), Release build, MSVC 19.44 on Windows 11, Python 3.14 with SciPy 1.16.3.
- Inputs: `results/mot17/detections/yolo26n/` (YOLO26n at 640, all 7 videos, 5,316 frames), and the first 40 clips (by file name) of the Anti-UAV test split from `results/antiuav/detections/`, once at 20% random deletion (`drop-0.2-seed0`) and once with 1 s gaps (`gap-1.0s`). Using test files is fine here: nothing was tuned on this comparison. The Anti-UAV files exercise the search gate, its cap and the duplicate merge far more than MOT17 does.
- Compared per frame: the list of (id, status, class, frames_since_seen, path length) must be equal; then the largest absolute difference over every `box` and `predicted_box` coordinate.
- The comparison was an ad-hoc script, not committed. The committed check, which also times the tracking step, is #10 (`tether-parity`). It has since run on 462 files with no mismatch: see [cpp-tracking-speed.md](cpp-tracking-speed.md).

## Results

| Input | Files | Track rows | IDs and statuses | Largest box difference (px) |
|---|---|---|---|---|
| MOT17 train, YOLO26n 640 | 7 | 92,546 | match on every frame | 2.6e-9 (MOT17-13) |
| Anti-UAV test, 20% deleted | 40 | not counted | match on every frame | 1.4e-10 |
| Anti-UAV test, 1 s gaps | 40 | not counted | match on every frame | 3.8e-10 |

The files are not byte-identical between the languages: the last digits of some boxes differ, as expected (see [floating-point differences](../CONCEPTS.md#floating-point-differences-between-languages)). The C++ output is byte-identical between Windows (MSVC) and Linux (GCC 13.3 under WSL) on MOT17-09 (the one file run on both), and a CLI test checks that two runs on the same input are byte-identical.

## What made it match

- **The same assignment solver, line for line.** Hungarian matching can have several equally cheap answers, and which one a solver returns depends on the order it scans rows and columns. The C++ port is a translation of SciPy's `linear_sum_assignment` (Crouse's shortest augmenting path), including its reverse column scan, its preference for a free column on a tie, and its transpose when there are more rows than columns. A test pins it to SciPy's answers on tie-heavy matrices.
- **The same loop orders everywhere else:** high detections before low ones, tracks in ID order, detections in file order, and the merge's nearest duplicate broken by track index as Python's `min` over tuples does.
- **The same number printing.** The C++ writer prints doubles the way Python's `json.dumps` does (shortest digits that read back exactly, fixed notation for exponents −4 to 15), so the files read alike even though the bytes aren't identical.

## Caveats

- 87 files is what we ran here. #10's check then covered all 91 Anti-UAV test clips under all 5 dropout conditions plus the 7 MOT17 videos, with no mismatch. Rare ties in inputs we haven't run still can't be ruled out.
- The Python tracker's `--single-target` diagnostic (#34) is not ported, by design, so it wasn't compared.
- MOT17 was run at 640 only, not at 1280×736.
