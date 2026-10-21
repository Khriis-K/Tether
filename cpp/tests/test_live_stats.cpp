// The stats panel's numbers: frame time, FPS, end-to-end delay and step times
// over the most recent frames, worked out as tether-timing does for a whole log.

#include <gtest/gtest.h>

#include <vector>

#include "tether/live_stats.hpp"

using namespace tether;

namespace {

// A frame that starts at `start` and spends `ms` on each step in turn.
StepTimes frame_at(double start, std::array<double, 5> ms) {
  StepTimes t{};
  Span* spans[] = {&t.capture, &t.prep, &t.detect, &t.track, &t.draw};
  double now = start;
  for (int i = 0; i < 5; ++i) {
    *spans[i] = {now, now + ms[i]};
    now += ms[i];
  }
  return t;
}

}  // namespace

TEST(P99, IsTheNearestRankValue) {
  EXPECT_DOUBLE_EQ(p99({5}), 5);
  EXPECT_DOUBLE_EQ(p99({3, 1, 2}), 3);  // under 100 values: the largest
  std::vector<double> hundred;
  for (int i = 1; i <= 100; ++i) hundred.push_back(i);
  EXPECT_DOUBLE_EQ(p99(hundred), 99);  // rank ceil(0.99 * 100) = 99
  hundred.push_back(101);
  EXPECT_DOUBLE_EQ(p99(hundred), 100);  // rank ceil(0.99 * 101) = 100
}

TEST(LiveStats, HasNothingToSayBeforeTheFirstFrame) {
  LiveStats stats;
  EXPECT_FALSE(stats.summary().has_value());
}

TEST(LiveStats, OneFrameGivesStepTimesAndDelayButNoFrameTime) {
  LiveStats stats;
  stats.add(frame_at(0, {2, 1, 10, 0.5, 1.5}));

  auto s = stats.summary();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->frames, 1u);
  EXPECT_FALSE(s->frame_time.has_value());
  EXPECT_FALSE(s->fps.has_value());
  EXPECT_DOUBLE_EQ(s->end_to_end.mean, 13);  // capture end (2) to draw end (15)
  EXPECT_DOUBLE_EQ(s->steps[0].mean, 2);
  EXPECT_DOUBLE_EQ(s->steps[2].mean, 10);
  EXPECT_DOUBLE_EQ(s->steps[4].mean, 1.5);
}

TEST(LiveStats, FrameTimeRunsFromOneDrawEndToTheNext) {
  LiveStats stats;
  stats.add(frame_at(0, {1, 1, 1, 1, 1}));   // draw ends at 5
  stats.add(frame_at(5, {1, 1, 1, 1, 1}));   // 10
  stats.add(frame_at(10, {1, 1, 16, 1, 1}));  // 30

  auto s = stats.summary();
  ASSERT_TRUE(s->frame_time.has_value());
  EXPECT_DOUBLE_EQ(s->frame_time->mean, 12.5);  // gaps of 5 and 20
  EXPECT_DOUBLE_EQ(s->frame_time->p99, 20);
  EXPECT_DOUBLE_EQ(*s->fps, 80);  // 1000 / 12.5
  EXPECT_DOUBLE_EQ(s->end_to_end.mean, 9);  // 5 - 1, 10 - 6 and 30 - 11
  EXPECT_DOUBLE_EQ(s->end_to_end.p99, 19);
  EXPECT_DOUBLE_EQ(s->steps[2].p99, 16);
}

TEST(LiveStats, KeepsOnlyTheMostRecentFrames) {
  LiveStats stats(3);
  stats.add(frame_at(0, {100, 100, 100, 100, 100}));  // slow warm-up, soon forgotten
  for (int i = 0; i < 3; ++i) stats.add(frame_at(1000 + 10.0 * i, {2, 2, 2, 2, 2}));

  auto s = stats.summary();
  EXPECT_EQ(s->frames, 3u);
  EXPECT_DOUBLE_EQ(s->frame_time->mean, 10);
  EXPECT_DOUBLE_EQ(s->steps[0].p99, 2);
  EXPECT_DOUBLE_EQ(s->end_to_end.mean, 8);
}

TEST(LiveStats, DefaultWindowIsTheLast300Frames) {
  LiveStats stats;
  for (int i = 0; i < 400; ++i) stats.add(frame_at(10.0 * i, {1, 1, 1, 1, 1}));
  EXPECT_EQ(stats.summary()->frames, 300u);
}
