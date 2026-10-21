// The frame prep setting: tether-pipeline --prep cpu or cuda, the prep each one
// makes, and the stats panel's frame prep row. Built with or without CUDA;
// test_cuda_frame_prep.cpp covers the kernel itself.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tether/frame_prep.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;

namespace {

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

}  // namespace

TEST(MakeFramePrep, CpuPrepLeavesTheBlobOnTheCpu) {
  auto prep = make_frame_prep(Prep::cpu, 736, 1280);

  auto prepared = prep->prepare(cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(7)));

  EXPECT_NE(dynamic_cast<CpuFramePrep*>(prep.get()), nullptr);
  EXPECT_EQ(prepared.blob.size(), 3u * 736 * 1280);
  EXPECT_EQ(prepared.gpu_blob, nullptr);
}

TEST(MakeFramePrep, CudaPrepLeavesTheBlobOnTheGpuOrSaysHowToBuildIt) {
  if (!cuda_prep_built()) {
    try {
      make_frame_prep(Prep::cuda, 736, 1280);
      FAIL() << "expected an error";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("TensorRT_ROOT"), std::string::npos) << e.what();
    }
    return;
  }
  auto prepared = make_frame_prep(Prep::cuda, 736, 1280)->prepare(cv::Mat(1080, 1920, CV_8UC3, cv::Scalar::all(7)));

  EXPECT_TRUE(prepared.blob.empty());
  EXPECT_NE(prepared.gpu_blob, nullptr);
  EXPECT_EQ(prepared.height, 736);
  EXPECT_EQ(prepared.width, 1280);
}

TEST(RunSettingsPrep, TheFramePrepRowSaysWhereFramesArePrepared) {
  EXPECT_EQ(run_settings("models/yolo26n.onnx").at(2), (std::pair<std::string, std::string>{"frame prep", "CPU"}));
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential, Runtime::tensorrt, true, Prep::cuda).at(2),
            (std::pair<std::string, std::string>{"frame prep", "CUDA (GPU)"}));
}

TEST(PipelineCliPrep, AnUnknownPrepIsABadArgument) {
  std::string err;

  EXPECT_EQ(cli({"clip.mp4", "--runtime", "tensorrt", "--prep", "opencl"}, &err), 2);
  EXPECT_NE(err.find("--prep: invalid value 'opencl'"), std::string::npos) << err;
}

TEST(PipelineCliPrep, CudaPrepOnlyAppliesToTensorRt) {
  // ONNX Runtime runs on the CPU, so a blob left on the GPU would have to come back.
  std::string err;

  EXPECT_EQ(cli({"clip.mp4", "--prep", "cuda"}, &err), 2);
  EXPECT_NE(err.find("--prep cuda"), std::string::npos) << err;
  EXPECT_NE(err.find("--runtime tensorrt"), std::string::npos) << err;
}

TEST(PipelineCliPrep, CpuPrepIsAllowedWithEitherRuntime) {
  // No such video: getting past the arguments to fail on the source (or the
  // model) is exit 1, where a bad argument would be 2.
  EXPECT_EQ(cli({"no-such-clip.mp4", "--model", "no-such-model.onnx", "--prep", "cpu"}), 1);
}
