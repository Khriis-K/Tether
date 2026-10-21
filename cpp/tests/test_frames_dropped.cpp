// frames_dropped edge case found by mutation testing (docs/spec-verify/issue-11.md).

#include <gtest/gtest.h>

#include "tether/video_source.hpp"

TEST(FramesDropped, AGapShorterThanHalfAPeriodIsNoneNotNegative) {
  // A frame can arrive early after a late one; that is jitter, not minus one drop.
  EXPECT_EQ(tether::frames_dropped(10.0, 33.3), 0);
}
