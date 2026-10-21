// Low-score pass: detections scored between LOW_SCORE and min_score only keep
// tracked tracks alive. Ported from python/tests/test_low_score.py.

#include <gtest/gtest.h>

#include "test_util.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

const Box BOX = {0.0, 50.0, 20.0, 70.0};      // a still object, so every prediction is this box
const Box SHIFTED = {8.0, 50.0, 28.0, 70.0};  // IoU 0.43 with BOX: matches by IoU, but not as a low detection
constexpr double HIGH = 0.9;
constexpr double LOW = 0.15;

struct Confirmed {
  Tracker tracker{{.confirm_frames = 3}};
  int id = 0;
};

Confirmed confirmed_tracker() {
  Confirmed c;
  std::vector<Track> tracks;
  for (int i = 0; i < 3; ++i) tracks = c.tracker.update({det(BOX, HIGH)});
  EXPECT_EQ(only(tracks).status, "tracked");
  c.id = tracks[0].id;
  return c;
}

}  // namespace

TEST(LowScore, LowDetectionKeepsATrackedTrackTracked) {
  auto c = confirmed_tracker();

  auto tracks = c.tracker.update({det(BOX, LOW)});
  const auto& t = only(tracks);

  EXPECT_EQ(t.id, c.id);
  EXPECT_EQ(t.status, "tracked");
  EXPECT_EQ(t.frames_since_seen, 0);
}

TEST(LowScore, LowDetectionBelowTheIouGateDoesNotKeepATrackTracked) {
  auto c = confirmed_tracker();

  auto tracks = c.tracker.update({det(SHIFTED, LOW)});
  const auto& t = only(tracks);

  EXPECT_EQ(t.id, c.id);
  EXPECT_EQ(t.status, "searching");
}

TEST(LowScore, HighDetectionAtTheSameOverlapStillMatches) {
  auto c = confirmed_tracker();

  auto tracks = c.tracker.update({det(SHIFTED, HIGH)});
  const auto& t = only(tracks);

  EXPECT_EQ(t.id, c.id);
  EXPECT_EQ(t.status, "tracked");
}

TEST(LowScore, DetectionAtOrBelowTheLowFloorIsIgnored) {
  auto c = confirmed_tracker();

  auto tracks = c.tracker.update({det(BOX, 0.1)});
  const auto& t = only(tracks);

  EXPECT_EQ(t.id, c.id);
  EXPECT_EQ(t.status, "searching");
}

TEST(LowScore, LowDetectionNeverStartsATrack) {
  Tracker tracker({.confirm_frames = 1});

  for (int i = 0; i < 5; ++i) EXPECT_TRUE(tracker.update({det(BOX, LOW)}).empty());
}

TEST(LowScore, LowDetectionDoesNotMatchASearchingTrack) {
  auto c = confirmed_tracker();
  c.tracker.update({});  // missed: now searching

  auto tracks = c.tracker.update({det(BOX, LOW)});
  const auto& t = only(tracks);

  EXPECT_EQ(t.id, c.id);
  EXPECT_EQ(t.status, "searching");
  EXPECT_EQ(t.frames_since_seen, 2);
}

TEST(LowScore, LowDetectionDoesNotMatchATentativeTrack) {
  Tracker tracker({.confirm_frames = 3});
  tracker.update({det(BOX, HIGH)});  // tentative

  EXPECT_TRUE(tracker.update({det(BOX, LOW)}).empty());  // the tentative track missed, so it was dropped
}
