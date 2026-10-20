#include "tether/video_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace tether {

int frames_dropped(double gap_ms, double period_ms) {
  if (gap_ms <= 0 || period_ms <= 0) return 0;
  return std::max(0, static_cast<int>(std::lround(gap_ms / period_ms)) - 1);
}

namespace {

// A file (or image sequence) read with OpenCV.
class FileSource : public VideoSource {
 public:
  explicit FileSource(cv::VideoCapture capture) : capture_(std::move(capture)) {}
  bool read(cv::Mat& frame) override { return capture_.read(frame); }
  std::optional<int> dropped() const override { return std::nullopt; }

 private:
  cv::VideoCapture capture_;
};

// A webcam read with OpenCV. Counts drops from the gaps between the camera's own
// frame timestamps, so they show even when the driver buffers frames.
class CameraSource : public VideoSource {
 public:
  explicit CameraSource(cv::VideoCapture capture) : capture_(std::move(capture)) {
    double fps = capture_.get(cv::CAP_PROP_FPS);
    period_ms_ = fps > 0 ? 1000.0 / fps : 0.0;
  }
  bool read(cv::Mat& frame) override {
    if (!capture_.read(frame)) return false;
    double stamp = capture_.get(cv::CAP_PROP_POS_MSEC);
    dropped_ = last_stamp_ ? frames_dropped(stamp - *last_stamp_, period_ms_) : 0;
    last_stamp_ = stamp;
    return true;
  }
  std::optional<int> dropped() const override { return dropped_; }

 private:
  cv::VideoCapture capture_;
  double period_ms_;
  std::optional<double> last_stamp_;
  int dropped_ = 0;
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
  return std::make_unique<FileSource>(std::move(capture));
}

}  // namespace tether
