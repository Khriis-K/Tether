// The pipeline: capture, prep, detect, track and draw, either one frame at a
// time (sequential) or as parallel stages joined by bounded queues. The
// sequential loop mirrors python/tether/demo.py (timed_tracking, dump_detections).
#pragma once

#include <cstddef>
#include <functional>
#include <opencv2/core.hpp>
#include <vector>

#include "tether/detector.hpp"
#include "tether/formats.hpp"
#include "tether/frame_prep.hpp"
#include "tether/tracker.hpp"
#include "tether/video_source.hpp"

namespace tether {

using Clock = std::function<double()>;  // milliseconds from a monotonic clock
// Draws and shows one frame's tracks; false once the run should stop.
using Show = std::function<bool(const cv::Mat& frame, const std::vector<Track>& tracks)>;

// std::chrono::steady_clock in milliseconds.
double steady_clock_ms();

// Run each frame through every step, calling `on_frame` with the frame's step
// times as it finishes. Stops when the source ends or `show` returns false
// (that frame is still reported). Each frame carries the source's dropped count.
void run_sequential(VideoSource& source, FramePrep& prep, Detector& detector, Tracker& tracker, const Show& show,
                    const Clock& clock, const std::function<void(const StepTimes&)>& on_frame);

// How many frames wait between two parallel stages. One is enough for the
// detector to find its next frame ready, and each extra frame waiting adds a
// detect time to the end-to-end delay: going from 2 to 1 kept FPS and cut the
// delay by about a third (docs/benchmarks/parallel-pipeline.md).
constexpr size_t PARALLEL_QUEUE_CAPACITY = 1;

// run_sequential with the steps overlapped: capture, prep, detect and track
// each run on their own thread, on different frames at once, and pass frames
// on through bounded queues. Draw and `on_frame` run on the caller's thread (a
// window must be drawn on the thread that made it), in frame order, so the
// tracker sees every frame in order and gives the same tracks as sequential.
// Ends as run_sequential does; frames read ahead are then dropped. An error in
// any step ends the run once the frames before it are drawn, and is rethrown.
void run_parallel(VideoSource& source, FramePrep& prep, Detector& detector, Tracker& tracker, const Show& show,
                  const Clock& clock, const std::function<void(const StepTimes&)>& on_frame);

// Every frame's detections, as they come, without tracking or filtering. Stops
// when the source ends or `keep_going` returns false (asked before each frame).
void dump_detections(VideoSource& source, FramePrep& prep, Detector& detector,
                     const std::function<void(const std::vector<Detection>&)>& on_frame,
                     const std::function<bool()>& keep_going);

}  // namespace tether
