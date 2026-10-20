// The sequential pipeline: one frame at a time through capture, prep, detect,
// track and draw. Mirrors python/tether/demo.py (timed_tracking, dump_detections).
#pragma once

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

// Every frame's detections, as they come, without tracking or filtering. Stops
// when the source ends or `keep_going` returns false (asked before each frame).
void dump_detections(VideoSource& source, FramePrep& prep, Detector& detector,
                     const std::function<void(const std::vector<Detection>&)>& on_frame,
                     const std::function<bool()>& keep_going);

}  // namespace tether
