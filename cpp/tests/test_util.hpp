// Helpers shared by the tests.
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "tether/formats.hpp"
#include "tether/tracker.hpp"

namespace tether::test {

// A path in a fresh directory for the running test, like pytest's tmp_path.
inline std::filesystem::path temp_path(const std::string& name) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  auto dir = std::filesystem::temp_directory_path() / "tether_tests" /
             (std::string(info->test_suite_name()) + "." + info->name());
  static std::string cleaned;  // wipe each test's directory once, on first use
  if (cleaned != dir.string()) {
    std::filesystem::remove_all(dir);
    cleaned = dir.string();
  }
  std::filesystem::create_directories(dir);
  return dir / name;
}

inline std::string read_bytes(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// One tracker run over scripted frames: each frame's tracks.
inline TrackFrames run(const DetectionFrames& frames, const TrackerSettings& settings = {}) {
  Tracker tracker(settings);
  TrackFrames outputs;
  for (const auto& dets : frames) outputs.push_back(tracker.update(dets));
  return outputs;
}

inline Detection det(const Box& box, double score = 0.9, int cls = 0) { return {box, score, cls}; }

// The one track a frame should hold.
inline const Track& only(const std::vector<Track>& tracks) {
  if (tracks.size() != 1) throw std::runtime_error("expected one track, got " + std::to_string(tracks.size()));
  return tracks[0];
}

inline const Track* find_id(const std::vector<Track>& tracks, int id) {
  for (const auto& t : tracks)
    if (t.id == id) return &t;
  return nullptr;
}

inline std::vector<int> ids(const std::vector<Track>& tracks) {
  std::vector<int> out;
  for (const auto& t : tracks) out.push_back(t.id);
  return out;
}

// Like pytest.approx: each coordinate within `abs` (default: relative 1e-6).
inline ::testing::AssertionResult boxes_near(const Box& actual, const Box& expected, double abs = -1) {
  for (size_t i = 0; i < 4; ++i) {
    double tol = abs >= 0 ? abs : 1e-6 * std::abs(expected[i]) + 1e-12;
    if (std::abs(actual[i] - expected[i]) > tol)
      return ::testing::AssertionFailure() << "coordinate " << i << ": " << actual[i] << " vs " << expected[i];
  }
  return ::testing::AssertionSuccess();
}

}  // namespace tether::test
