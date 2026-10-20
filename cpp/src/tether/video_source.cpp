#include "tether/video_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <utility>

namespace tether {

int frames_dropped(double gap_ms, double period_ms) {
  if (gap_ms <= 0 || period_ms <= 0) return 0;
  return std::max(0, static_cast<int>(std::lround(gap_ms / period_ms)) - 1);
}

std::optional<int> DropCounter::add(double stamp_ms) {
  if (period_ms_ <= 0 || stamp_ms <= 0) return std::nullopt;
  std::optional<double> last = std::exchange(last_stamp_, stamp_ms);
  return last ? frames_dropped(stamp_ms - *last, period_ms_) : 0;
}

namespace {

// A file (or image sequence) read with OpenCV.
class FileSource : public VideoSource {
 public:
  FileSource(cv::VideoCapture capture, bool sequence) : capture_(std::move(capture)), sequence_(sequence) {}
  bool read(cv::Mat& frame) override { return capture_.read(frame); }
  std::optional<int> dropped() const override { return std::nullopt; }
  // An image sequence has no rate of its own (FFmpeg makes up 25), so it can't tell.
  double fps() const override { return sequence_ ? 0 : std::max(0.0, capture_.get(cv::CAP_PROP_FPS)); }

 private:
  cv::VideoCapture capture_;
  bool sequence_;
};

// A webcam read with OpenCV. Counts drops from the gaps between the camera's own
// frame timestamps, so they show even when the driver buffers frames.
class CameraSource : public VideoSource {
 public:
  explicit CameraSource(cv::VideoCapture capture)
      : capture_(std::move(capture)), drops_(period_ms(capture_.get(cv::CAP_PROP_FPS))) {}
  bool read(cv::Mat& frame) override {
    if (!capture_.read(frame)) return false;
    dropped_ = drops_.add(capture_.get(cv::CAP_PROP_POS_MSEC));
    return true;
  }
  std::optional<int> dropped() const override { return dropped_; }
  double fps() const override { return std::max(0.0, capture_.get(cv::CAP_PROP_FPS)); }

 private:
  static double period_ms(double fps) { return fps > 0 ? 1000.0 / fps : 0.0; }

  cv::VideoCapture capture_;
  DropCounter drops_;
  std::optional<int> dropped_;
};

}  // namespace

std::unique_ptr<VideoSource> open_video_source(const std::string& source) {
  bool webcam = !source.empty() && std::all_of(source.begin(), source.end(), [](unsigned char c) {
    return std::isdigit(c);
  });
  cv::VideoCapture capture;
  if (webcam)
    capture.open(std::stoi(source));
  else
    capture.open(source);
  if (!capture.isOpened()) throw std::runtime_error("can't open video source '" + source + "'");
  if (webcam) return std::make_unique<CameraSource>(std::move(capture));
  return std::make_unique<FileSource>(std::move(capture), source.find('%') != std::string::npos);
}

}  // namespace tether
