// Detection and track files, shared with the Python tracker.
//
// Both are JSON Lines, one line per frame, in frame order. See docs/FORMATS.md.
#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace tether {

using Box = std::array<double, 4>;  // x1, y1, x2, y2 in pixels
using Point = std::array<double, 2>;

struct Detection {
  Box box;
  double score;
  int cls;

  bool operator==(const Detection&) const = default;
};

struct Track {
  int id;
  Box box;
  int cls;
  std::string status;  // "tentative", "tracked" or "searching"
  Box predicted_box;
  int frames_since_seen;
  std::vector<Point> path;  // recent box centers, oldest first

  bool operator==(const Track&) const = default;
};

using DetectionFrames = std::vector<std::vector<Detection>>;
using TrackFrames = std::vector<std::vector<Track>>;

// Loads throw std::runtime_error on a missing file or a frame out of order.
void save_detections(const std::filesystem::path& path, const DetectionFrames& frames);
DetectionFrames load_detections(const std::filesystem::path& path);
void save_tracks(const std::filesystem::path& path, const TrackFrames& frames);
TrackFrames load_tracks(const std::filesystem::path& path);

// One frame of a timing log: when each pipeline step started and ended.
using Span = std::array<double, 2>;  // [start, end] in milliseconds from a steady clock
struct StepTimes {
  Span capture, prep, detect, track, draw;
  std::optional<int> dropped;  // frames the camera dropped just before this one; webcam runs only
};

// Writes a detection file one frame at a time, flushing each line, so a run cut
// short (a webcam stopped with Ctrl-C) still leaves a valid file. Throws
// std::runtime_error when the file can't be opened or written.
class DetectionWriter {
 public:
  explicit DetectionWriter(const std::filesystem::path& path);
  void write(const std::vector<Detection>& detections);

 private:
  std::filesystem::path path_;
  std::ofstream file_;
  size_t frame_ = 0;
};

// The same for a timing log (docs/FORMATS.md, "Timing log").
class TimingLogWriter {
 public:
  explicit TimingLogWriter(const std::filesystem::path& path);
  void write(const StepTimes& times);

 private:
  std::filesystem::path path_;
  std::ofstream file_;
  size_t frame_ = 0;
};

}  // namespace tether
