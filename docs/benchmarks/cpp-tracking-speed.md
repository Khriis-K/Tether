# C++ vs. Python: matching tracks and tracking-step speed (#10)

**Question:** does the committed match check (`tether-parity`) pass on real detection files, and how much faster is the C++ tracking step than the Python one?

**Answer:** it passes on all 462 real files we have (the 7 MOT17 videos and all 91 Anti-UAV test clips under 5 dropout conditions) and on the 7 scripted scenes. On MOT17 the C++ tracking step is **14.1× faster** (2.52 → 0.18 ms per frame). On Anti-UAV it's **35.0× faster** (0.172 → 0.0049 ms per frame). Both clear the spec's "10x+" for the tracking step. That target is about the tracking step only, not the whole pipeline.

## Setup

- `python -m tether.parity` at commit 715950d, both trackers at their defaults. Windows 11, AMD Ryzen AI 7 350, 31 GB RAM, on mains power. C++: Release build, MSVC 19.44 (Visual Studio 17 2022 generator). Python 3.14.0 with NumPy 2.3.4 and SciPy 1.16.3.
- **What's timed:** a fresh default tracker's `update` over every frame of the file, end to end, including building each frame's output tracks. File reading and writing are left out on both sides. Python times in-process with `time.perf_counter`. C++ times in `tether-time-track` with `std::chrono::steady_clock`. Each file runs 5 times and the median is kept. A speedup over several files is total Python time over total C++ time, so long files weigh more.
- **Match:** every frame must have the same tracks in the same order, with the same ID, status, class, `frames_since_seen` and path length, and every box, predicted box and path point within 1e-6 px.
- Inputs: `results/mot17/detections/yolo26n/` (YOLO26n at 640, 5,316 frames), and `results/antiuav/detections/{drop-0.0-seed0, drop-0.2-seed0, gap-0.25s, gap-0.5s, gap-1.0s}`, each the 91-clip Anti-UAV test split (85,374 frames per condition). Nothing was tuned on these runs, so using the test split is fine. Results files: `results/parity/mot17-scripted.json` and `results/parity/antiuav-<condition>.json` (not committed; `results/` is ignored).

## Results

| Input | Files | Frames | All match | Largest box difference (px) | Python ms/frame | C++ ms/frame | Speedup (total) | Per-file range |
|---|---|---|---|---|---|---|---|---|
| MOT17 train, YOLO26n 640 | 7 | 5,316 | yes | 2.6e-9 | 2.521 | 0.1785 | **14.1×** | 11.7–17.2× |
| Anti-UAV test, no deletion | 91 | 85,374 | yes | 5.3e-10 | 0.182 | 0.0054 | 33.7× | 22.0–44.7× |
| Anti-UAV test, 20% deleted | 91 | 85,374 | yes | 5.8e-10 | 0.170 | 0.0048 | 35.6× | 25.2–62.8× |
| Anti-UAV test, 0.25 s gaps | 91 | 85,374 | yes | 5.3e-10 | 0.176 | 0.0052 | 33.8× | 24.5–64.0× |
| Anti-UAV test, 0.5 s gaps | 91 | 85,374 | yes | 6.0e-10 | 0.164 | 0.0047 | 35.0× | 25.5–73.2× |
| Anti-UAV test, 1 s gaps | 91 | 85,374 | yes | 6.2e-10 | 0.170 | 0.0046 | 37.1× | 22.5–52.6× |
| Anti-UAV, all 5 conditions | 455 | 426,870 | yes | 6.2e-10 | 0.172 | 0.0049 | **35.0×** | 22.0–73.2× |
| Scripted scenes (`--scripted`) | 7 | 275 | yes | 2.8e-14 | | | | 11.7–48.9× |

Per MOT17 video (median of 5, whole file): MOT17-02 15.2×, -04 13.3×, -05 17.2×, -09 13.3×, -10 13.4×, -11 11.7×, -13 14.4×. The scripted scenes are a few dozen frames each, so their times are too short to quote as speed numbers. They're there to check that the outputs match.

## Reading the numbers

- **Why Anti-UAV gains more than MOT17 (a hypothesis, not measured):** an Anti-UAV frame has one or two tracks, so Python's time goes mostly to fixed per-frame costs: building small NumPy arrays and calling SciPy and the interpreter, where C++ has almost none. A MOT17 frame has dozens of tracks, so more of Python's time is in NumPy's compiled matrix work, which C++ can't beat by as much. Profiling would settle it; we haven't run one.
- **This is one step, not the pipeline.** In the Python pipeline the detector costs far more than tracking, so the whole-pipeline speedup (user story 34, a later issue) will be much smaller. See [Amdahl's law](../CONCEPTS.md#amdahls-law).
- **Repeat spread:** the medians hide some run-to-run noise. The MOT17 total moved from 13,295 to 13,402 ms in Python and from 946 to 949 ms in C++ between an earlier run and this one (14.05× vs. 14.12×).

## Caveats

- One machine, one compiler, one run of 5 repeats per file. Laptop CPUs boost and throttle, so treat the second digit loosely.
- The C++ timer runs in a new process, so its first repeat is cold. Python's runs straight after an untimed pass over the same file, so it's warm. The median of 5 hides the one cold repeat, but `ctest`'s `--repeats 1` run doesn't, and its numbers aren't speed results.
- "Same tracks" is checked to 1e-6 px. The two languages don't produce byte-identical files (see [floating-point differences](../CONCEPTS.md#floating-point-differences-between-languages)).
- The Python tracker's `--single-target` diagnostic isn't ported, so it isn't compared.
- MOT17 was run at 640 only.
