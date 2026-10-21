// The TensorRT detector's CUDA graph: the engine captured once and replayed each
// frame must give the same detections as launching it directly. Uses the
// brightness stand-in (tests/onnx/make_models.py), whose score is the frame's
// mean value, so a replay that reused an old frame's input or output would show.
// Only built with TensorRT.

#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"
#include "tether/tensorrt_detector.hpp"

using namespace tether;
using tether::test::read_bytes;
using tether::test::temp_path;

namespace {

const std::string ENGINE = std::string(TETHER_TEST_ENGINES) + "/brightness_736x1280.engine";
const std::string MODEL = std::string(TETHER_TEST_DATA) + "/brightness_736x1280.onnx";
const std::vector<int> GREYS = {60, 200, 120, 255};  // a different score each frame

std::vector<std::vector<Detection>> run(TensorRtDetector& detector) {
  CpuFramePrep prep(736, 1280);
  std::vector<std::vector<Detection>> out;
  for (int grey : GREYS) out.push_back(detector.detect(prep.prepare(cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(grey)))));
  return out;
}

}  // namespace

TEST(CudaGraph, TheEngineIsCapturedByDefault) {
  TensorRtDetector detector(ENGINE);

  EXPECT_TRUE(detector.uses_cuda_graph());
}

TEST(CudaGraph, CanBeTurnedOff) {
  TensorRtDetector detector(ENGINE, DETECTOR_MIN_SCORE, false);

  EXPECT_FALSE(detector.uses_cuda_graph());
}

TEST(CudaGraph, GivesTheSameDetectionsAsLaunchingDirectlyFrameAfterFrame) {
  TensorRtDetector graph(ENGINE), direct(ENGINE, DETECTOR_MIN_SCORE, false);

  auto a = run(graph), b = run(direct);

  ASSERT_EQ(a.size(), GREYS.size());
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i].size(), 1u) << "frame " << i;
    ASSERT_EQ(b[i].size(), 1u) << "frame " << i;
    EXPECT_EQ(a[i][0].box, b[i][0].box) << "frame " << i;
    EXPECT_EQ(a[i][0].score, b[i][0].score) << "frame " << i;
    EXPECT_EQ(a[i][0].cls, b[i][0].cls) << "frame " << i;
  }
  // Each replay ran on its own frame: brighter frames score higher.
  EXPECT_LT(a[0][0].score, a[2][0].score);
  EXPECT_LT(a[2][0].score, a[1][0].score);
  EXPECT_LT(a[1][0].score, a[3][0].score);
}

TEST(PipelineCliCudaGraph, DumpsTheSameDetectionsWithAndWithoutTheGraph) {
  auto model = temp_path("brightness.onnx");
  std::filesystem::copy_file(MODEL, model);
  std::filesystem::copy_file(ENGINE, temp_path("brightness.engine"));
  auto dir = model.parent_path();
  for (size_t i = 0; i < GREYS.size(); ++i)
    cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(GREYS[i])));
  auto graph = temp_path("graph.jsonl"), direct = temp_path("direct.jsonl");
  std::ostringstream out, err;
  std::vector<std::string> common = {(dir / "f%d.png").string(), "--model", model.string(), "--runtime", "tensorrt"};

  auto with = common, without = common;
  with.insert(with.end(), {"--dump", graph.string()});
  without.insert(without.end(), {"--no-cuda-graph", "--dump", direct.string()});
  ASSERT_EQ(run_pipeline_cli(with, out, err), 0) << err.str();
  ASSERT_EQ(run_pipeline_cli(without, out, err), 0) << err.str();

  EXPECT_FALSE(read_bytes(graph).empty());
  EXPECT_EQ(read_bytes(graph), read_bytes(direct));
}
