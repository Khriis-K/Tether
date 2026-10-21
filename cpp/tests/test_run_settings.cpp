// The speed settings the stats panel says produced its numbers.

#include <gtest/gtest.h>

#include "tether/pipeline_cli.hpp"

using namespace tether;

TEST(RunSettings, NameTheModelRuntimeFramePrepAndPipelineMode) {
  Settings expected{{"model", "yolo26s.onnx"},
                    {"runtime", "ONNX Runtime (CPU)"},
                    {"frame prep", "CPU"},
                    {"pipeline", "sequential, 1 thread"}};
  EXPECT_EQ(run_settings("models/yolo26s.onnx"), expected);
}
