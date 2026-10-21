// Counting a webcam's dropped frames from its frame timestamps, and saying
// "unknown" instead of 0 when the camera doesn't give usable ones (code review, #11).

#include <gtest/gtest.h>

#include <optional>

#include "tether/video_source.hpp"

using tether::DropCounter;

TEST(DropCounter, TheFirstFrameHasNoneDroppedBeforeIt) {
  DropCounter counter(1000.0 / 30);

  EXPECT_EQ(counter.add(5000.0), 0);
}

TEST(DropCounter, CountsFromTheGapSinceThePreviousFrame) {
  DropCounter counter(1000.0 / 30);
  counter.add(5000.0);

  EXPECT_EQ(counter.add(5033.3), 0);
  EXPECT_EQ(counter.add(5166.7), 3);  // four periods later
}

TEST(DropCounter, UnknownWhenTheCameraReportsNoFrameRate) {
  DropCounter counter(0.0);

  EXPECT_EQ(counter.add(5000.0), std::nullopt);
  EXPECT_EQ(counter.add(5133.3), std::nullopt);
}

TEST(DropCounter, UnknownWhenTheCameraGivesNoTimestamps) {
  // Some backends report 0 (or -1) for every frame; a gap of 0 would read as "no drops".
  DropCounter counter(1000.0 / 30);

  EXPECT_EQ(counter.add(0.0), std::nullopt);
  EXPECT_EQ(counter.add(-1.0), std::nullopt);
}
