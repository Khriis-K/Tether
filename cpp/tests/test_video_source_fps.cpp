// The rate a video source plays at, which --record saves the video at.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "test_util.hpp"
#include "tether/video_source.hpp"

using namespace tether;
using tether::test::temp_path;

TEST(VideoSourceFps, AVideoFileGivesItsRate) {
  auto path = temp_path("v.avi");
  {
    cv::VideoWriter w(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12, {64, 48});
    for (int i = 0; i < 2; ++i) w.write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(0, 0, 0)));
  }
  EXPECT_DOUBLE_EQ(open_video_source(path.string())->fps(), 12);
}

TEST(VideoSourceFps, AnImageSequenceCantTell) {
  // MOT17 is stored this way: JPEGs, which FFmpeg reads at a made-up 25 FPS.
  auto dir = temp_path("x").parent_path();
  for (int i = 1; i <= 2; ++i)
    cv::imwrite((dir / cv::format("%06d.jpg", i)).string(), cv::Mat(48, 64, CV_8UC3, cv::Scalar(0, 0, 0)));
  EXPECT_EQ(open_video_source((dir / "%06d.jpg").string())->fps(), 0);
}
