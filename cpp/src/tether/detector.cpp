#include "tether/detector.hpp"

#include <algorithm>
#include <array>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>

#ifdef TETHER_HAS_TENSORRT
#include "tether/tensorrt_detector.hpp"
#endif

namespace tether {

std::vector<Detection> decode(const std::vector<float>& rows, const PreparedFrame& p, double min_score) {
  // Numpy does this in float32 (a float32 array combined with Python numbers), then
  // clips against the frame size; mirroring that keeps the boxes bit-equal.
  auto threshold = static_cast<float>(min_score);
  auto scale = static_cast<float>(p.scale);
  auto to_frame = [&](float v, int pad, int limit) {
    float mapped = (v - static_cast<float>(pad)) / scale;
    return std::min(std::max(static_cast<double>(mapped), 0.0), static_cast<double>(limit));
  };
  std::vector<Detection> out;
  for (size_t i = 0; i + 6 <= rows.size(); i += 6) {
    const float* r = &rows[i];
    if (r[4] < threshold) continue;
    double x1 = to_frame(r[0], p.pad_x, p.frame_width), x2 = to_frame(r[2], p.pad_x, p.frame_width);
    double y1 = to_frame(r[1], p.pad_y, p.frame_height), y2 = to_frame(r[3], p.pad_y, p.frame_height);
    if (x2 <= x1 || y2 <= y1) continue;
    out.push_back({{x1, y1, x2, y2}, static_cast<double>(r[4]), static_cast<int>(r[5])});
  }
  return out;
}

std::pair<int, int> model_input_size(const std::vector<int64_t>& shape) {
  bool fixed = shape.size() == 4 && std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d > 0; });
  if (!fixed) {
    std::string dims;
    for (auto d : shape) dims += (dims.empty() ? "" : ", ") + (d > 0 ? std::to_string(d) : std::string("?"));
    throw std::invalid_argument("model input shape [" + dims + "] must be a fixed [1, 3, height, width]; "
                                "export with dynamic=False");
  }
  return {static_cast<int>(shape[2]), static_cast<int>(shape[3])};
}

struct OnnxDetector::Session {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tether"};
  Ort::Session session{nullptr};
  std::string input_name, output_name;
  std::pair<int, int> input_size;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

OnnxDetector::OnnxDetector(const std::filesystem::path& model, double min_score)
    : session_(std::make_unique<Session>()), min_score_(min_score) {
  try {
    Ort::SessionOptions options;  // the CPU provider, ONNX Runtime's default
    session_->session = Ort::Session(session_->env, model.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    session_->input_name = session_->session.GetInputNameAllocated(0, allocator).get();
    session_->output_name = session_->session.GetOutputNameAllocated(0, allocator).get();
    auto shape = session_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    session_->input_size = model_input_size(shape);
  } catch (const std::exception& e) {  // Ort::Exception or std::invalid_argument
    throw std::runtime_error(model.string() + ": " + e.what());
  }
}

OnnxDetector::~OnnxDetector() = default;

std::pair<int, int> OnnxDetector::input_size() const { return session_->input_size; }

std::vector<Detection> OnnxDetector::detect(const PreparedFrame& prepared) {
  if (prepared.blob.empty())
    throw std::invalid_argument("ONNX Runtime runs on the CPU and needs CPU frame prep; this frame's blob is on the GPU");
  std::array<int64_t, 4> shape = {1, 3, prepared.height, prepared.width};
  // ONNX Runtime wants a mutable pointer but only reads an input tensor.
  auto input = Ort::Value::CreateTensor<float>(session_->memory, const_cast<float*>(prepared.blob.data()),
                                               prepared.blob.size(), shape.data(), shape.size());
  const char* input_name = session_->input_name.c_str();
  const char* output_name = session_->output_name.c_str();
  auto outputs = session_->session.Run(Ort::RunOptions{nullptr}, &input_name, &input, 1, &output_name, 1);
  const float* data = outputs[0].GetTensorData<float>();
  std::vector<float> rows(data, data + outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());
  return decode(rows, prepared, min_score_);
}

bool tensorrt_built() {
#ifdef TETHER_HAS_TENSORRT
  return true;
#else
  return false;
#endif
}

std::filesystem::path engine_for_model(const std::filesystem::path& model) {
  return std::filesystem::path(model).replace_extension(".engine");
}

std::unique_ptr<Detector> make_detector(Runtime runtime, const std::filesystem::path& model, double min_score,
                                        [[maybe_unused]] bool cuda_graph) {
  if (runtime == Runtime::onnxruntime) return std::make_unique<OnnxDetector>(model, min_score);
#ifdef TETHER_HAS_TENSORRT
  return std::make_unique<TensorRtDetector>(engine_for_model(model), min_score, cuda_graph);
#else
  throw std::runtime_error("this build has no TensorRT runtime; configure CMake with -DTensorRT_ROOT=<TensorRT folder>");
#endif
}

}  // namespace tether
