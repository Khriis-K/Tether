// The TensorRT detector, on the stand-in model's engine (built by CMake with
// trtexec from tests/onnx/fixed_736x1280.onnx). Only built with TensorRT.
// Agreement with ONNX Runtime on the real model is tether-runtime-compare's job.

#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"
#include "tether/tensorrt_detector.hpp"

using namespace tether;
using tether::test::boxes_near;
using tether::test::read_bytes;
using tether::test::temp_path;

namespace {

const std::string ENGINE = std::string(TETHER_TEST_ENGINES) + "/fixed_736x1280.engine";
const std::string MODEL = std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx";

}  // namespace

TEST(TensorRtDetector, ReadsItsInputSizeFromTheEngine) {
  TensorRtDetector detector(ENGINE);

  EXPECT_EQ(detector.input_size(), (std::pair<int, int>{736, 1280}));
}

TEST(TensorRtDetector, GivesTheSameDetectionsAsOnnxRuntime) {
  TensorRtDetector trt(ENGINE);
  OnnxDetector ort(MODEL);
  cv::Mat frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
  auto prepared = CpuFramePrep(736, 1280).prepare(frame);

  auto a = trt.detect(prepared), b = ort.detect(prepared);

  ASSERT_EQ(a.size(), 1u);
  ASSERT_EQ(b.size(), 1u);
  EXPECT_TRUE(boxes_near(a[0].box, {150.0, 150.0, 450.0, 450.0}, 1e-3));
  EXPECT_EQ(a[0].box, b[0].box);
  EXPECT_EQ(a[0].score, b[0].score);
  EXPECT_EQ(a[0].cls, b[0].cls);
}

TEST(TensorRtDetector, RejectsAFrameOfAnotherSize) {
  TensorRtDetector detector(ENGINE);
  cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

  EXPECT_THROW(detector.detect(CpuFramePrep(640, 640).prepare(frame)), std::invalid_argument);
}

TEST(TensorRtDetector, AMissingEngineSaysHowToBuildOne) {
  try {
    TensorRtDetector detector(temp_path("none.engine"));
    FAIL() << "expected an error";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("trtexec --onnx="), std::string::npos) << e.what();
  }
}

TEST(TensorRtDetector, AFileThatIsntAnEngineIsAnError) {
  // An ONNX model where its engine should be: TensorRT can't deserialize it.
  EXPECT_THROW(TensorRtDetector(std::filesystem::path(MODEL)), std::runtime_error);
}

TEST(PipelineCliRuntime, TensorRtDumpsTheSameDetectionsAsOnnxRuntime) {
  // The engine sits next to "its" model, as engine_for_model expects.
  auto model = temp_path("stand_in.onnx");
  std::filesystem::copy_file(MODEL, model);
  std::filesystem::copy_file(ENGINE, temp_path("stand_in.engine"));
  cv::Mat black(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
  auto dir = model.parent_path();
  for (int i = 0; i < 3; ++i) cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), black);
  auto ort = temp_path("ort.jsonl"), trt = temp_path("trt.jsonl");
  std::ostringstream out, err;

  ASSERT_EQ(run_pipeline_cli({(dir / "f%d.png").string(), "--model", model.string(), "--dump", ort.string()}, out, err),
            0)
      << err.str();
  ASSERT_EQ(run_pipeline_cli({(dir / "f%d.png").string(), "--model", model.string(), "--runtime", "tensorrt", "--dump",
                              trt.string()},
                             out, err),
            0)
      << err.str();

  EXPECT_FALSE(read_bytes(trt).empty());
  EXPECT_EQ(read_bytes(trt), read_bytes(ort));
}
