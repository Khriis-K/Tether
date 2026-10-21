// The runtime setting: tether-pipeline --runtime onnxruntime or tensorrt, the
// detector each one makes, and where TensorRT looks for its engine. Built with or
// without TensorRT; test_tensorrt_detector.cpp covers the detector itself.

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "tether/detector.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;

namespace {

const std::string MODEL = std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx";

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

}  // namespace

TEST(EngineForModel, IsTheModelPathEndingInDotEngine) {
  EXPECT_EQ(engine_for_model("models/yolo26n_1280x736.onnx"), std::filesystem::path("models/yolo26n_1280x736.engine"));
}

TEST(MakeDetector, OnnxRuntimeRunsTheOnnxModel) {
  auto detector = make_detector(Runtime::onnxruntime, MODEL);

  EXPECT_NE(dynamic_cast<OnnxDetector*>(detector.get()), nullptr);
  EXPECT_EQ(detector->input_size(), (std::pair<int, int>{736, 1280}));
}

TEST(MakeDetector, TensorRtLooksForTheEngineNextToTheModel) {
  // There's no fixed_736x1280.engine in the test data, so this fails either way:
  // without TensorRT built in, or for want of the engine.
  try {
    make_detector(Runtime::tensorrt, MODEL);
    FAIL() << "expected an error";
  } catch (const std::runtime_error& e) {
    std::string what = e.what();
    if (tensorrt_built()) {
      EXPECT_NE(what.find("fixed_736x1280.engine"), std::string::npos) << what;
      EXPECT_NE(what.find("trtexec"), std::string::npos) << what;
    } else {
      EXPECT_NE(what.find("TensorRT_ROOT"), std::string::npos) << what;
    }
  }
}

TEST(RunSettingsRuntime, TheRuntimeRowNamesTheRuntimeAndWhereItRuns) {
  EXPECT_EQ(run_settings("models/yolo26n.onnx").at(1),
            (std::pair<std::string, std::string>{"runtime", "ONNX Runtime (CPU)"}));
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential, Runtime::tensorrt).at(1),
            (std::pair<std::string, std::string>{"runtime", "TensorRT (GPU)"}));
}

TEST(PipelineCliRuntime, AnUnknownRuntimeIsABadArgument) {
  std::string err;

  EXPECT_EQ(cli({"clip.mp4", "--runtime", "openvino"}, &err), 2);
  EXPECT_NE(err.find("--runtime: invalid value 'openvino'"), std::string::npos) << err;
}

TEST(PipelineCliRuntime, ATensorRtRunWithNoEngineFailsAndSaysWhy) {
  std::string err;

  EXPECT_EQ(cli({"clip.mp4", "--model", MODEL, "--runtime", "tensorrt"}, &err), 1);
  EXPECT_NE(err.find(tensorrt_built() ? "trtexec" : "TensorRT_ROOT"), std::string::npos) << err;
}
