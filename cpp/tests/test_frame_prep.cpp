// CPU frame prep: letterbox into the model's (height, width), mirroring python/tether/detector.py.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "tether/frame_prep.hpp"

using namespace tether;

TEST(Letterbox, ScalesTheLongSideToFitAndPadsTheShortSideEvenly) {
  auto g = letterbox_geometry(480, 640, 640, 640);

  EXPECT_EQ(g.scale, 1.0);
  EXPECT_EQ(g.width, 640);
  EXPECT_EQ(g.height, 480);
  EXPECT_EQ(g.pad_x, 0);
  EXPECT_EQ(g.pad_y, 80);
}

TEST(Letterbox, ShrinksFramesBiggerThanTheModelInput) {
  auto g = letterbox_geometry(720, 1280, 640, 640);

  EXPECT_EQ(g.scale, 0.5);
  EXPECT_EQ(g.pad_x, 0);
  EXPECT_EQ(g.pad_y, 140);
}

TEST(Letterbox, FitsAWideFrameIntoARectangularInput) {
  auto g = letterbox_geometry(1080, 1920, 736, 1280);

  EXPECT_DOUBLE_EQ(g.scale, 2.0 / 3.0);
  EXPECT_EQ(g.width, 1280);
  EXPECT_EQ(g.height, 720);
  EXPECT_EQ(g.pad_x, 0);
  EXPECT_EQ(g.pad_y, 8);
}

TEST(Letterbox, PadsTheSidesWhenTheFrameIsNarrowerThanTheInput) {
  auto g = letterbox_geometry(736, 640, 736, 1280);

  EXPECT_EQ(g.scale, 1.0);
  EXPECT_EQ(g.pad_x, 320);
  EXPECT_EQ(g.pad_y, 0);
}

TEST(Letterbox, RoundsHalvesToEvenLikePython) {
  // 5 px wide at scale 0.5 is 2.5 px: Python's round() gives 2, std::round would give 3.
  auto g = letterbox_geometry(1280, 5, 640, 640);

  EXPECT_EQ(g.width, 2);
  EXPECT_EQ(g.pad_x, 319);
}

TEST(Letterbox, PadsARectangularInputTopAndBottomWithGray) {
  cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(7, 7, 7));
  auto g = letterbox_geometry(1080, 1920, 736, 1280);

  cv::Mat image = letterbox(frame, g, 736, 1280);

  ASSERT_EQ(image.rows, 736);
  ASSERT_EQ(image.cols, 1280);
  ASSERT_EQ(image.type(), CV_8UC3);
  EXPECT_EQ(cv::countNonZero(image.rowRange(0, 8).reshape(1) != PAD_VALUE), 0);
  EXPECT_EQ(cv::countNonZero(image.rowRange(8, 728).reshape(1) != 7), 0);
  EXPECT_EQ(cv::countNonZero(image.rowRange(728, 736).reshape(1) != PAD_VALUE), 0);
}

TEST(CpuFramePrep, BuildsABlobOfTheRequestedHeightAndWidth) {
  cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));

  auto prepared = CpuFramePrep(736, 1280).prepare(frame);

  EXPECT_EQ(prepared.blob.size(), 3u * 736 * 1280);
  EXPECT_EQ(prepared.height, 736);
  EXPECT_EQ(prepared.width, 1280);
  EXPECT_DOUBLE_EQ(prepared.scale, 2.0 / 3.0);
  EXPECT_EQ(prepared.pad_x, 0);
  EXPECT_EQ(prepared.pad_y, 8);
  EXPECT_EQ(prepared.frame_height, 1080);
  EXPECT_EQ(prepared.frame_width, 1920);
}

TEST(CpuFramePrep, BlobIsRgbChannelsFirstScaledToOne) {
  // A 1x2 BGR frame into a 2x2 input: row 0 is the frame, row 1 is padding.
  cv::Mat frame(1, 2, CV_8UC3);
  frame.at<cv::Vec3b>(0, 0) = {10, 20, 30};  // B, G, R
  frame.at<cv::Vec3b>(0, 1) = {40, 50, 60};

  auto blob = CpuFramePrep(2, 2).prepare(frame).blob;

  auto at = [&](int c, int y, int x) { return blob[c * 4 + y * 2 + x]; };
  EXPECT_EQ(at(0, 0, 0), 30.0f / 255.0f);  // R first
  EXPECT_EQ(at(1, 0, 0), 20.0f / 255.0f);
  EXPECT_EQ(at(2, 0, 0), 10.0f / 255.0f);
  EXPECT_EQ(at(0, 0, 1), 60.0f / 255.0f);
  EXPECT_EQ(at(2, 0, 1), 40.0f / 255.0f);
  EXPECT_EQ(at(0, 1, 0), 114.0f / 255.0f);  // padding
  EXPECT_EQ(at(2, 1, 1), 114.0f / 255.0f);
}
