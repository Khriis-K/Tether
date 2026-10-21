// CUDA frame prep feeding the TensorRT detector straight from GPU memory, with
// and without the CUDA graph. Uses the brightness stand-in (tests/onnx/make_models.py),
// whose score is the input's mean value. Its frames are flat gray, which every
// resize leaves exactly as it is, so CPU and CUDA prep give the same input and
// the detections must be identical. A replay that read an old frame would show
// as the wrong score. Only built with TensorRT.

#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/cuda_frame_prep.hpp"
#include "tether/pipeline_cli.hpp"
#include "tether/tensorrt_detector.hpp"

using namespace tether;
using tether::test::read_bytes;
using tether::test::temp_path;

namespace {

const std::string ENGINE = std::string(TETHER_TEST_ENGINES) + "/brightness_736x1280.engine";
const std::string MODEL = std::string(TETHER_TEST_DATA) + "/brightness_736x1280.onnx";
const std::vector<int> GREYS = {60, 200, 120, 255};

cv::Mat grey(int value) { return cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(value)); }

std::vector<std::vector<Detection>> run(TensorRtDetector& detector, FramePrep& prep) {
  std::vector<std::vector<Detection>> out;
  for (int g : GREYS) out.push_back(detector.detect(prep.prepare(grey(g))));
  return out;
}

void expect_same(const std::vector<std::vector<Detection>>& a, const std::vector<std::vector<Detection>>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i].size(), 1u) << "frame " << i;
    ASSERT_EQ(b[i].size(), 1u) << "frame " << i;
    EXPECT_EQ(a[i][0].box, b[i][0].box) << "frame " << i;
    EXPECT_EQ(a[i][0].score, b[i][0].score) << "frame " << i;
    EXPECT_EQ(a[i][0].cls, b[i][0].cls) << "frame " << i;
  }
}

}  // namespace

TEST(CudaPrepDetector, GivesTheSameDetectionsAsCpuPrepThroughTheGraph) {
  TensorRtDetector detector(ENGINE);
  ASSERT_TRUE(detector.uses_cuda_graph());
  CudaFramePrep cuda(736, 1280);
  CpuFramePrep cpu(736, 1280);

  auto a = run(detector, cuda), b = run(detector, cpu);

  expect_same(a, b);
  // Each replay read its own frame: brighter frames score higher.
  EXPECT_LT(a[0][0].score, a[2][0].score);
  EXPECT_LT(a[2][0].score, a[1][0].score);
  EXPECT_LT(a[1][0].score, a[3][0].score);
}

TEST(CudaPrepDetector, GivesTheSameDetectionsAsCpuPrepWithoutTheGraph) {
  TensorRtDetector detector(ENGINE, DETECTOR_MIN_SCORE, false);
  CudaFramePrep cuda(736, 1280);
  CpuFramePrep cpu(736, 1280);

  expect_same(run(detector, cuda), run(detector, cpu));
}

TEST(CudaPrepDetector, TensorRtRejectsAGpuFrameOfAnotherSize) {
  TensorRtDetector detector(ENGINE);

  EXPECT_THROW(detector.detect(CudaFramePrep(640, 640).prepare(grey(60))), std::invalid_argument);
}

TEST(CudaPrepDetector, OnnxRuntimeRejectsAGpuFrameAndSaysWhy) {
  OnnxDetector detector(MODEL);

  try {
    detector.detect(CudaFramePrep(736, 1280).prepare(grey(60)));
    FAIL() << "expected an error";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("CPU frame prep"), std::string::npos) << e.what();
  }
}

TEST(PipelineCliPrep, CudaPrepDumpsTheSameDetectionsAsCpuPrep) {
  auto model = temp_path("brightness.onnx");
  std::filesystem::copy_file(MODEL, model);
  std::filesystem::copy_file(ENGINE, temp_path("brightness.engine"));
  auto dir = model.parent_path();
  for (size_t i = 0; i < GREYS.size(); ++i) cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), grey(GREYS[i]));
  auto cpu = temp_path("cpu.jsonl"), cuda = temp_path("cuda.jsonl");
  std::ostringstream out, err;
  std::vector<std::string> common = {(dir / "f%d.png").string(), "--model", model.string(), "--runtime", "tensorrt"};

  auto with_cpu = common, with_cuda = common;
  with_cpu.insert(with_cpu.end(), {"--dump", cpu.string()});
  with_cuda.insert(with_cuda.end(), {"--prep", "cuda", "--dump", cuda.string()});
  ASSERT_EQ(run_pipeline_cli(with_cpu, out, err), 0) << err.str();
  ASSERT_EQ(run_pipeline_cli(with_cuda, out, err), 0) << err.str();

  EXPECT_FALSE(read_bytes(cpu).empty());
  EXPECT_EQ(read_bytes(cuda), read_bytes(cpu));
}

TEST(PipelineCliPrep, CudaPrepTracksInParallelMode) {
  auto model = temp_path("brightness.onnx");
  std::filesystem::copy_file(MODEL, model);
  std::filesystem::copy_file(ENGINE, temp_path("brightness.engine"));
  auto dir = model.parent_path();
  for (int i = 0; i < 8; ++i) cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), grey(60 + 20 * i));
  auto log = temp_path("timing.jsonl");
  std::ostringstream out, err;

  ASSERT_EQ(run_pipeline_cli({(dir / "f%d.png").string(), "--model", model.string(), "--runtime", "tensorrt", "--prep",
                              "cuda", "--pipeline", "parallel", "--timing", log.string()},
                             out, err),
            0)
      << err.str();

  EXPECT_NE(out.str().find("8 frames"), std::string::npos) << out.str();
}
