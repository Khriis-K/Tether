// CUDA frame prep against CPU frame prep: the same frame through both must give
// the same model input, value for value, within TOLERANCE. Only built with the
// CUDA toolkit (CMake given TensorRT_ROOT).
//
// The tolerance is one 8-bit gray level (1/255) per value. The CPU path resizes
// with OpenCV, whose 8-bit bilinear resize rounds in fixed point; the kernel
// copies its weights and its vectorized rounding, which gives bit-identical
// blobs for 1920x1080 frames, but OpenCV rounds the last few pixels of some
// rows its scalar way, so odd frame sizes can be a level off. Anything past
// one level is a real bug (a wrong source pixel, weight, channel or plane).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "tether/cuda_frame_prep.hpp"
#include "tether/frame_prep.hpp"

using namespace tether;

namespace {

// One gray level, plus float rounding: 101/255 - 100/255 is 0.0039215982 in
// float, a hair over 1/255 itself.
constexpr float TOLERANCE = 1.0f / 255.0f + 1e-6f;

cv::Mat noise(int rows, int cols, int seed = 1) {
  cv::Mat frame(rows, cols, CV_8UC3);
  cv::RNG rng(seed);
  rng.fill(frame, cv::RNG::UNIFORM, 0, 256);
  return frame;
}

// A sample frame that looks more like video than noise does: smooth shading
// with sharp-edged shapes on it, a different picture in each channel.
cv::Mat scene(int rows, int cols) {
  cv::Mat frame(rows, cols, CV_8UC3);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      frame.at<cv::Vec3b>(y, x) = {static_cast<uchar>(x * 255 / cols), static_cast<uchar>(y * 255 / rows),
                                   static_cast<uchar>((x + y) % 256)};
  cv::circle(frame, {cols / 3, rows / 2}, rows / 5, {20, 200, 240}, -1);
  cv::rectangle(frame, cv::Rect(cols / 2, rows / 4, cols / 7, rows / 9), {250, 10, 90}, -1);
  cv::line(frame, {0, rows - 1}, {cols - 1, 0}, {0, 0, 0}, 3);
  return frame;
}

// The largest difference between the two preps' blobs for `frame`.
float max_difference(CudaFramePrep& cuda, CpuFramePrep& cpu, const cv::Mat& frame) {
  auto want = cpu.prepare(frame).blob;
  auto got = download_blob(cuda.prepare(frame));
  EXPECT_EQ(got.size(), want.size());
  float worst = 0;
  for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
  return worst;
}

float max_difference(const cv::Mat& frame, int height, int width) {
  CudaFramePrep cuda(height, width);
  CpuFramePrep cpu(height, width);
  return max_difference(cuda, cpu, frame);
}

}  // namespace

TEST(CudaFramePrep, MatchesCpuOnANoiseFrameIntoTheNonSquareInput) {
  // 1920x1080 into 1280x736: shrunk by 2/3, gray bars top and bottom.
  EXPECT_LE(max_difference(noise(1080, 1920), 736, 1280), TOLERANCE);
}

TEST(CudaFramePrep, MatchesCpuOnANoiseFrameIntoTheSquareInput) {
  EXPECT_LE(max_difference(noise(1080, 1920, 2), 640, 640), TOLERANCE);
}

TEST(CudaFramePrep, MatchesCpuOnASampleScene) {
  auto frame = scene(1080, 1920);

  EXPECT_LE(max_difference(frame, 736, 1280), TOLERANCE);
  EXPECT_LE(max_difference(frame, 640, 640), TOLERANCE);
}

TEST(CudaFramePrep, MatchesCpuWhenEnlargingANarrowFrame) {
  // 480x360 into 1280x736: scaled up by 736/480, gray bars left and right.
  EXPECT_LE(max_difference(noise(480, 360, 3), 736, 1280), TOLERANCE);
}

TEST(CudaFramePrep, MatchesCpuOnOddSizes) {
  EXPECT_LE(max_difference(noise(1280, 5, 4), 640, 640), TOLERANCE);  // 2 px wide once shrunk
  EXPECT_LE(max_difference(noise(333, 517, 5), 736, 1280), TOLERANCE);
}

TEST(CudaFramePrep, AFrameThatNeedsNoResizeIsCopiedExactly) {
  // 1280x720 already fits 1280x736: only padding, so no rounding at all.
  EXPECT_EQ(max_difference(noise(720, 1280, 6), 736, 1280), 0.0f);
}

TEST(CudaFramePrep, MatchesCpuOnAFrameThatIsPartOfABiggerImage) {
  // A region of interest: its rows aren't back to back in memory.
  cv::Mat big = noise(1200, 2100, 7);
  cv::Mat frame = big(cv::Rect(90, 60, 1920, 1080));
  ASSERT_FALSE(frame.isContinuous());

  EXPECT_LE(max_difference(frame, 736, 1280), TOLERANCE);
}

TEST(CudaFramePrep, KeepsMatchingWhenTheFrameSizeChanges) {
  CudaFramePrep cuda(736, 1280);
  CpuFramePrep cpu(736, 1280);

  EXPECT_LE(max_difference(cuda, cpu, noise(1080, 1920, 8)), TOLERANCE);
  EXPECT_LE(max_difference(cuda, cpu, noise(480, 640, 9)), TOLERANCE);
  EXPECT_LE(max_difference(cuda, cpu, noise(1080, 1920, 10)), TOLERANCE);
}

TEST(CudaFramePrep, GivesTheSameGeometryAsCpu) {
  cv::Mat frame = noise(1080, 1920);

  auto got = CudaFramePrep(736, 1280).prepare(frame);
  auto want = CpuFramePrep(736, 1280).prepare(frame);

  EXPECT_EQ(got.height, want.height);
  EXPECT_EQ(got.width, want.width);
  EXPECT_EQ(got.scale, want.scale);
  EXPECT_EQ(got.pad_x, want.pad_x);
  EXPECT_EQ(got.pad_y, want.pad_y);
  EXPECT_EQ(got.frame_height, want.frame_height);
  EXPECT_EQ(got.frame_width, want.frame_width);
}

TEST(CudaFramePrep, LeavesTheBlobOnTheGpu) {
  auto prepared = CudaFramePrep(736, 1280).prepare(noise(1080, 1920));

  EXPECT_TRUE(prepared.blob.empty());
  EXPECT_NE(prepared.gpu_blob, nullptr);
  EXPECT_EQ(download_blob(prepared).size(), 3u * 736 * 1280);
}

TEST(CudaFramePrep, EachPreparedFrameKeepsItsOwnBlob) {
  // In parallel mode the next frame is prepared while the detector still reads
  // the one before, so a new frame mustn't reuse a buffer that's still held.
  CudaFramePrep cuda(640, 640);
  CpuFramePrep cpu(640, 640);
  cv::Mat a = noise(1080, 1920, 11), b = noise(1080, 1920, 12);

  auto first = cuda.prepare(a);
  auto second = cuda.prepare(b);
  auto first_blob = download_blob(first), want = cpu.prepare(a).blob;

  ASSERT_EQ(first_blob.size(), want.size());
  float worst = 0;
  for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(first_blob[i] - want[i]));
  EXPECT_LE(worst, TOLERANCE);
  EXPECT_NE(first.gpu_blob.get(), second.gpu_blob.get());
}

TEST(CudaFramePrep, DownloadingACpuPreparedFrameGivesItsBlob) {
  auto prepared = CpuFramePrep(2, 2).prepare(cv::Mat(1, 2, CV_8UC3, cv::Scalar(10, 20, 30)));

  EXPECT_EQ(download_blob(prepared), prepared.blob);
}
