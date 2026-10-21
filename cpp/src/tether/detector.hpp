// The detector: one prepared frame in, that frame's detections out.
//
// Runs a YOLO26 model exported to ONNX (tools/export_model.py). YOLO26 has no
// NMS step: the model outputs at most 300 final boxes as rows of
// [x1, y1, x2, y2, score, class] in model-input pixels. All that's left is to
// undo the letterbox and drop low scores. Mirrors python/tether/detector.py.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "tether/formats.hpp"
#include "tether/frame_prep.hpp"

namespace tether {

inline constexpr double DETECTOR_MIN_SCORE = 0.1;  // low on purpose: detection files are saved unfiltered

// Turn the model's output rows (6 floats each) into detections in frame pixels.
// Boxes are clipped to the frame; any left with no area are dropped. The math is
// in float32, as Python's numpy does it, so both pipelines write the same boxes.
std::vector<Detection> decode(const std::vector<float>& rows, const PreparedFrame& prepared, double min_score);

// (height, width) from an ONNX input shape [1, 3, height, width]. Throws
// std::invalid_argument for a dynamic (-1) or otherwise unusable shape.
std::pair<int, int> model_input_size(const std::vector<int64_t>& shape);

class Detector {
 public:
  virtual ~Detector() = default;
  // (height, width) of the model input that `detect` expects.
  virtual std::pair<int, int> input_size() const = 0;
  virtual std::vector<Detection> detect(const PreparedFrame& prepared) = 0;
};

// YOLO26 on ONNX Runtime's CPU provider. The provider is fixed so timings mean
// the same thing on every machine, as in the Python detector.
class OnnxDetector : public Detector {
 public:
  // Throws std::runtime_error when the model can't be loaded or its input isn't fixed.
  explicit OnnxDetector(const std::filesystem::path& model, double min_score = DETECTOR_MIN_SCORE);
  ~OnnxDetector() override;
  std::pair<int, int> input_size() const override;
  std::vector<Detection> detect(const PreparedFrame& prepared) override;

 private:
  struct Session;
  std::unique_ptr<Session> session_;
  double min_score_;
};

// Which runtime runs the model: ONNX Runtime on the CPU, or TensorRT on the GPU.
enum class Runtime { onnxruntime, tensorrt };

// Whether this build has the TensorRT runtime (CMake was given TensorRT_ROOT).
bool tensorrt_built();

// The TensorRT engine built from `model`: the same path ending in .engine.
std::filesystem::path engine_for_model(const std::filesystem::path& model);

// The detector for `model` on `runtime`; TensorRT loads engine_for_model(model).
// Throws std::runtime_error when the model or engine can't be loaded, or for
// TensorRT in a build without it.
std::unique_ptr<Detector> make_detector(Runtime runtime, const std::filesystem::path& model,
                                        double min_score = DETECTOR_MIN_SCORE);

}  // namespace tether
