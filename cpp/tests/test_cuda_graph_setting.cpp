// The CUDA graph setting: tether-pipeline --no-cuda-graph, and the stats panel's
// runtime row saying when the graph is off. Built with or without TensorRT;
// test_cuda_graph.cpp covers the graph itself.

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "tether/pipeline_cli.hpp"

using namespace tether;

TEST(RunSettingsCudaGraph, TheRuntimeRowSaysWhenTensorRtRunsWithoutTheGraph) {
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential, Runtime::tensorrt).at(1),
            (std::pair<std::string, std::string>{"runtime", "TensorRT (GPU)"}));
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential, Runtime::tensorrt, false).at(1),
            (std::pair<std::string, std::string>{"runtime", "TensorRT (GPU, no CUDA graph)"}));
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential, Runtime::onnxruntime).at(1),
            (std::pair<std::string, std::string>{"runtime", "ONNX Runtime (CPU)"}));
}

TEST(PipelineCliCudaGraph, NoCudaGraphOnlyAppliesToTensorRt) {
  std::ostringstream out, err;

  EXPECT_EQ(run_pipeline_cli({"clip.mp4", "--no-cuda-graph"}, out, err), 2);
  EXPECT_NE(err.str().find("--no-cuda-graph"), std::string::npos) << err.str();
  EXPECT_NE(err.str().find("--runtime tensorrt"), std::string::npos) << err.str();
}

TEST(PipelineCliCudaGraph, NoCudaGraphTakesNoValue) {
  std::ostringstream out, err;

  EXPECT_EQ(run_pipeline_cli({"clip.mp4", "--runtime", "tensorrt", "--no-cuda-graph=1"}, out, err), 2);
}
