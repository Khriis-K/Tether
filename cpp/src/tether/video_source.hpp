// The video source: frames from a webcam or a video file, one at a time.
//
// Neither mode sleeps. A file is decoded as fast as the reader can go, so the
// pipeline runs uncapped; a webcam read blocks until the camera's next frame, so
// webcam mode follows the camera's rate. Mirrors python/tether/video.py.
#pragma once

#include <memory>
#include <optional>
#include <opencv2/core.hpp>
#include <string>

namespace tether {

class VideoSource {
 public:
  virtual ~VideoSource() = default;
  // The next BGR frame, or false once the source has ended.
  virtual bool read(cv::Mat& frame) = 0;
  // Frames the camera dropped just before the last frame read: ones it captured
  // but the pipeline never got, because it was busy. Empty for a file, which
  // can't drop frames, and for a camera that can't tell.
  virtual std::optional<int> dropped() const = 0;
  // The rate the source plays at, in frames per second; 0 when it can't tell.
  virtual double fps() const { return 0; }
};

// Frames a camera dropped between two frames it delivered `gap_ms` apart, at a
// nominal `period_ms` per frame: the gap in whole periods, less the one expected.
// Rounds, so normal timing jitter doesn't count as a drop.
int frames_dropped(double gap_ms, double period_ms);

// Counts a camera's dropped frames from the timestamps it gives each frame.
class DropCounter {
 public:
  explicit DropCounter(double period_ms) : period_ms_(period_ms) {}
  // Frames dropped just before the frame stamped `stamp_ms`: 0 for the first
  // frame, empty when the camera gives no frame rate or no timestamps (<= 0),
  // so an uncountable camera never reads as one that dropped nothing.
  std::optional<int> add(double stamp_ms);

 private:
  double period_ms_;
  std::optional<double> last_stamp_;
};

// `source` is a webcam index ("0") or a video file path (or an image-sequence
// pattern such as "img1/%06d.jpg"). Opens right away, so opening a camera isn't
// timed as the first frame's capture. Throws std::runtime_error if it can't open.
std::unique_ptr<VideoSource> open_video_source(const std::string& source);

}  // namespace tether
