// Detection and track files, shared with the Python tracker.
//
// Both are JSON Lines, one line per frame, in frame order. See docs/FORMATS.md.
#pragma once

#include <array>
#include <filesystem>
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

}  // namespace tether
