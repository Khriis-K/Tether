// Track lifecycle: tentative -> tracked -> searching -> removed.
// Ported from python/tests/test_lifecycle.py.

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

constexpr double SIZE = 20.0;

Box box_at(double x, double y = 50.0) { return {x, y, x + SIZE, y + SIZE}; }

Box moving(int f, double speed = 3.0) { return box_at(speed * f); }

bool in(int f, int start, int stop) { return start <= f && f < stop; }

// One object moving steadily, undetected on frames [gap_start, gap_stop).
DetectionFrames with_gap(int n_frames, int gap_start, int gap_stop) {
  DetectionFrames frames;
  for (int f = 0; f < n_frames; ++f)
    frames.push_back(in(f, gap_start, gap_stop) ? std::vector<Detection>{} : std::vector<Detection>{det(moving(f))});
  return frames;
}

}  // namespace

TEST(Lifecycle, NewDetectionStartsAsATentativeTrack) {
  auto tracks = run({{det(box_at(0.0))}}, {.confirm_frames = 3})[0];

  EXPECT_EQ(only(tracks).status, "tentative");
}

TEST(Lifecycle, TrackIsConfirmedAfterTheConfirmThresholdInFrames) {
  for (int confirm_frames : {1, 2, 3, 5}) {
    SCOPED_TRACE("confirm_frames " + std::to_string(confirm_frames));
    DetectionFrames frames;
    for (int f = 0; f < 8; ++f) frames.push_back({det(moving(f))});

    auto outputs = run(frames, {.confirm_frames = confirm_frames});

    std::set<int> seen;
    for (int f = 0; f < 8; ++f) {
      EXPECT_EQ(only(outputs[f]).status, f < confirm_frames - 1 ? "tentative" : "tracked") << "frame " << f;
      seen.insert(outputs[f][0].id);
    }
    EXPECT_EQ(seen.size(), 1u);
  }
}

TEST(Lifecycle, OneFrameFalseDetectionNeverBecomesAConfirmedTrack) {
  // A real object throughout, plus a blip far away on frame 5 only.
  DetectionFrames frames;
  for (int f = 0; f < 15; ++f) frames.push_back({det(moving(f))});
  frames[5].push_back(det(box_at(400.0, 300.0)));

  auto outputs = run(frames, {.confirm_frames = 3, .max_missing_frames = 30});

  int real_id = outputs[0][0].id;
  ASSERT_EQ(outputs[5].size(), 2u);
  const auto& blip = outputs[5][0].id == real_id ? outputs[5][1] : outputs[5][0];
  EXPECT_NE(blip.id, real_id);
  EXPECT_EQ(blip.status, "tentative");
  for (size_t f = 6; f < outputs.size(); ++f) EXPECT_EQ(find_id(outputs[f], blip.id), nullptr) << "frame " << f;
  for (const auto& tracks : outputs)
    if (const auto* real = find_id(tracks, real_id)) EXPECT_NE(real->status, "searching");
}

TEST(Lifecycle, TentativeTrackThatMissesAFrameIsDroppedBeforeConfirming) {
  auto outputs = run({{det(box_at(0.0))}, {det(box_at(0.0))}, {}, {det(box_at(0.0))}},
                     {.confirm_frames = 3, .max_missing_frames = 30});

  EXPECT_TRUE(outputs[2].empty());
  EXPECT_NE(outputs[3][0].id, outputs[0][0].id);
  EXPECT_EQ(outputs[3][0].status, "tentative");
}

TEST(Lifecycle, ConfirmedTrackThatLosesItsDetectionIsSearchingAndCountsMissedFrames) {
  auto outputs = run(with_gap(15, 10, 15), {.confirm_frames = 3, .max_missing_frames = 30});

  int track_id = outputs[0][0].id;
  for (int f = 10, k = 1; f < 15; ++f, ++k) {
    EXPECT_EQ(ids(outputs[f]), std::vector<int>{track_id});
    EXPECT_EQ(outputs[f][0].status, "searching");
    EXPECT_EQ(outputs[f][0].frames_since_seen, k);
  }
}

TEST(Lifecycle, SearchingTrackKeepsPredictingAlongItsMotionAndKeepsItsLastSeenBox) {
  auto outputs = run(with_gap(16, 10, 16), {.confirm_frames = 3, .max_missing_frames = 30});

  for (int f = 10; f < 16; ++f) {
    const auto& track = outputs[f][0];
    EXPECT_EQ(track.box, moving(9));
    // Predicted box for the next frame follows the object's 3 px/frame.
    EXPECT_TRUE(boxes_near(track.predicted_box, moving(f + 1), 1.0)) << "frame " << f;
  }
}

TEST(Lifecycle, ObjectThatDisappearsForFewerFramesThanTheLimitKeepsItsId) {
  for (int gap_length : {1, 5, 12}) {
    SCOPED_TRACE("gap " + std::to_string(gap_length));
    auto outputs = run(with_gap(10 + gap_length + 5, 10, 10 + gap_length),
                       {.confirm_frames = 3, .max_missing_frames = 12});

    int track_id = outputs[0][0].id;
    for (const auto& tracks : outputs) EXPECT_EQ(ids(tracks), std::vector<int>{track_id});
    const auto& back = outputs[10 + gap_length][0];
    EXPECT_EQ(back.status, "tracked");
    EXPECT_EQ(back.frames_since_seen, 0);
    EXPECT_EQ(back.box, moving(10 + gap_length));
  }
}

TEST(Lifecycle, StationaryObjectThatDisappearsAndReturnsKeepsItsId) {
  DetectionFrames frames;
  for (int i = 0; i < 10; ++i) frames.push_back({det(box_at(100.0))});
  for (int i = 0; i < 8; ++i) frames.push_back({});
  for (int i = 0; i < 5; ++i) frames.push_back({det(box_at(100.0))});

  auto outputs = run(frames, {.confirm_frames = 3, .max_missing_frames = 12});

  for (const auto& tracks : outputs) EXPECT_EQ(ids(tracks), std::vector<int>{outputs[0][0].id});
}

TEST(Lifecycle, TrackIsRemovedOnceMissingLongerThanTheLimit) {
  for (int max_missing_frames : {1, 5, 12}) {
    SCOPED_TRACE("max_missing_frames " + std::to_string(max_missing_frames));
    int gap_stop = 10 + max_missing_frames + 1;
    auto outputs = run(with_gap(10 + max_missing_frames + 3, 10, gap_stop),
                       {.confirm_frames = 3, .max_missing_frames = max_missing_frames});

    int track_id = outputs[0][0].id;
    int last_searching = 10 + max_missing_frames - 1;
    EXPECT_EQ(outputs[last_searching][0].id, track_id);
    EXPECT_EQ(outputs[last_searching][0].frames_since_seen, max_missing_frames);
    EXPECT_TRUE(outputs[last_searching + 1].empty());
    // The object comes back, but its old track is gone: it's a new object now.
    const auto& returned = outputs[gap_stop][0];
    EXPECT_NE(returned.id, track_id);
    EXPECT_EQ(returned.status, "tentative");
  }
}

TEST(Lifecycle, TwoObjectsCrossingWhileOneHidesTheOtherDoNotSwapIds) {
  // Same lane, 6 px/frame toward each other, meeting around frame 20. While
  // they overlap, only the front one is detected; the hidden one is searching.
  auto rightward = [](int f) { return box_at(6.0 * f, 50.0); };
  auto leftward = [](int f) { return box_at(240.0 - 6.0 * f, 54.0); };
  auto hidden = [](int f) { return in(f, 17, 24); };
  DetectionFrames frames;
  for (int f = 0; f < 40; ++f) {
    frames.push_back({det(rightward(f))});
    if (!hidden(f)) frames.back().push_back(det(leftward(f)));
  }

  auto outputs = run(frames, {.confirm_frames = 3, .max_missing_frames = 30});

  auto id_with_box = [](const std::vector<Track>& tracks, const Box& box) {
    for (const auto& t : tracks)
      if (t.box == box) return t.id;
    return -1;
  };
  int right_id = id_with_box(outputs[0], rightward(0));
  int left_id = id_with_box(outputs[0], leftward(0));
  EXPECT_NE(right_id, left_id);
  for (int f = 0; f < 40; ++f) {
    EXPECT_EQ(id_with_box(outputs[f], rightward(f)), right_id) << "frame " << f;
    if (!hidden(f)) {
      EXPECT_EQ(id_with_box(outputs[f], leftward(f)), left_id) << "frame " << f;
    } else {
      const auto* left = find_id(outputs[f], left_id);
      ASSERT_NE(left, nullptr) << "frame " << f;
      EXPECT_EQ(left->status, "searching") << "frame " << f;
    }
  }
}

TEST(Lifecycle, DefaultSettingsConfirmAndRemoveTracks) {
  Tracker tracker;
  std::vector<std::string> statuses;
  for (int i = 0; i < 10; ++i) statuses.push_back(tracker.update({det(box_at(0.0))})[0].status);
  EXPECT_EQ(statuses.front(), "tentative");
  EXPECT_EQ(statuses.back(), "tracked");

  TrackFrames outputs;
  for (int i = 0; i < 1000; ++i) outputs.push_back(tracker.update({}));
  EXPECT_EQ(outputs.front()[0].status, "searching");
  EXPECT_TRUE(outputs.back().empty());
}
