// The video source: files (and image sequences) through one interface, and
// counting the frames a camera drops. A live webcam is checked by hand.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

#include "test_util.hpp"
#include "tether/video_source.hpp"

using namespace tether;
using tether::test::temp_path;

TEST(FramesDropped, NoneWhenFramesArriveOnePeriodApart) { EXPECT_EQ(frames_dropped(33.3, 33.3), 0); }

TEST(FramesDropped, AGapOfFourPeriodsIsThreeDroppedFrames) {
  // Measured on the dev laptop's 30 FPS camera: a 200 ms stall left a 133.4 ms gap.
  EXPECT_EQ(frames_dropped(133.4, 1000.0 / 30), 3);
}

TEST(FramesDropped, TimingJitterRoundsToTheNearestWholeFrame) {
  EXPECT_EQ(frames_dropped(1.4 * 33.3, 33.3), 0);
  EXPECT_EQ(frames_dropped(1.6 * 33.3, 33.3), 1);
}

TEST(FramesDropped, NoneWithoutUsableTimes) {
  EXPECT_EQ(frames_dropped(0.0, 33.3), 0);
  EXPECT_EQ(frames_dropped(-33.3, 33.3), 0);
  EXPECT_EQ(frames_dropped(100.0, 0.0), 0);  // a camera that reports no frame rate
}

TEST(VideoSource, ReadsAFileFrameByFrameUntilItEnds) {
  auto dir = temp_path("seq").parent_path();
  for (int i = 0; i < 3; ++i)
    cv::imwrite((dir / ("frame" + std::to_string(i) + ".png")).string(),
                cv::Mat(4, 6, CV_8UC3, cv::Scalar(10 * i, 0, 0)));

  auto source = open_video_source((dir / "frame%d.png").string());

  cv::Mat frame;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(source->read(frame)) << "frame " << i;
    EXPECT_EQ(frame.size(), cv::Size(6, 4));
    EXPECT_EQ(frame.at<cv::Vec3b>(0, 0)[0], 10 * i);  // in order
  }
  EXPECT_FALSE(source->read(frame));
}

TEST(VideoSource, AFileNeverReportsDroppedFrames) {
  auto dir = temp_path("seq").parent_path();
  cv::imwrite((dir / "frame0.png").string(), cv::Mat(4, 6, CV_8UC3, cv::Scalar(0, 0, 0)));
  auto source = open_video_source((dir / "frame%d.png").string());

  cv::Mat frame;
  ASSERT_TRUE(source->read(frame));

  EXPECT_FALSE(source->dropped().has_value());
}

TEST(VideoSource, AMissingFileIsAnError) {
  EXPECT_THROW(open_video_source(temp_path("missing.mp4").string()), std::runtime_error);
}
