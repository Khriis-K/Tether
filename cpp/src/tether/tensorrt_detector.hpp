// The TensorRT detector: the same YOLO26 model as OnnxDetector, run on the GPU
// from a TensorRT engine built out of the ONNX file (README.md, "TensorRT runtime").
// Only built when CMake is given TensorRT_ROOT.
#pragma once

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "tether/detector.hpp"

namespace tether {

// An engine is built for one GPU, one TensorRT version and one input size, so
// it isn't portable: build it on the machine that runs it, once per model.
class TensorRtDetector : public Detector {
 public:
  // Throws std::runtime_error when the engine is missing, won't load (built for
  // another GPU or TensorRT version), or isn't one fixed [1, 3, h, w] float
  // input and one [1, N, 6] float output.
  explicit TensorRtDetector(const std::filesystem::path& engine, double min_score = DETECTOR_MIN_SCORE);
  ~TensorRtDetector() override;
  std::pair<int, int> input_size() const override;
  // Copies the prepared frame to the GPU, runs the engine and copies the rows
  // back, all inside this call: its time is the whole model run.
  std::vector<Detection> detect(const PreparedFrame& prepared) override;

 private:
  struct Engine;
  std::unique_ptr<Engine> engine_;
  double min_score_;
};

}  // namespace tether
