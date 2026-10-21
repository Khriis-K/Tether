#include "tether/tensorrt_detector.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace tether {

namespace {

// TensorRT's messages, warnings and up, to stderr.
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << "\n";
  }
};

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::vector<int64_t> dims_of(const nvinfer1::Dims& d) { return {d.d, d.d + d.nbDims}; }

size_t elements(const nvinfer1::Dims& d) {
  size_t n = 1;
  for (int i = 0; i < d.nbDims; ++i) n *= static_cast<size_t>(d.d[i]);
  return n;
}

}  // namespace

struct TensorRtDetector::Engine {
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  void* input = nullptr;  // device buffers
  void* output = nullptr;
  size_t input_elements = 0, output_elements = 0;
  std::pair<int, int> input_size;

  ~Engine() {
    context.reset();  // before the buffers it points at
    cudaFree(input);
    cudaFree(output);
    if (stream) cudaStreamDestroy(stream);
  }

  void load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("no engine; build one from the ONNX model with trtexec --onnx=<model>.onnx "
                               "--saveEngine=<model>.engine --noTF32");
    std::vector<char> plan((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) throw std::runtime_error("can't start TensorRT");
    engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine)
      throw std::runtime_error("can't load the engine; an engine only runs on the GPU and TensorRT version "
                               "that built it, so rebuild it here with trtexec");

    const char *input_name = nullptr, *output_name = nullptr;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      const char* name = engine->getIOTensorName(i);
      if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(std::string("tensor '") + name + "' isn't float32");
      auto& slot = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? input_name : output_name;
      if (slot) throw std::runtime_error("expected one input and one output");
      slot = name;
    }
    if (!input_name || !output_name) throw std::runtime_error("expected one input and one output");
    auto in = engine->getTensorShape(input_name), out = engine->getTensorShape(output_name);
    try {
      input_size = model_input_size(dims_of(in));
    } catch (const std::invalid_argument& e) {
      throw std::runtime_error(e.what());
    }
    if (out.nbDims != 3 || out.d[0] != 1 || out.d[1] < 1 || out.d[2] != 6)
      throw std::runtime_error("output must be [1, N, 6] rows of [x1, y1, x2, y2, score, class]");
    input_elements = elements(in);
    output_elements = elements(out);

    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("can't create an execution context");
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    check(cudaMalloc(&input, input_elements * sizeof(float)), "cudaMalloc");
    check(cudaMalloc(&output, output_elements * sizeof(float)), "cudaMalloc");
    if (!context->setTensorAddress(input_name, input) || !context->setTensorAddress(output_name, output))
      throw std::runtime_error("can't bind the engine's tensors");
  }
};

TensorRtDetector::TensorRtDetector(const std::filesystem::path& engine, double min_score)
    : engine_(std::make_unique<Engine>()), min_score_(min_score) {
  try {
    engine_->load(engine);
  } catch (const std::exception& e) {
    throw std::runtime_error(engine.string() + ": " + e.what());
  }
}

TensorRtDetector::~TensorRtDetector() = default;

std::pair<int, int> TensorRtDetector::input_size() const { return engine_->input_size; }

std::vector<Detection> TensorRtDetector::detect(const PreparedFrame& prepared) {
  auto& e = *engine_;
  if (prepared.blob.size() != e.input_elements)
    throw std::invalid_argument("prepared frame is " + std::to_string(prepared.height) + "x" +
                                std::to_string(prepared.width) + ", but the engine takes " +
                                std::to_string(e.input_size.first) + "x" + std::to_string(e.input_size.second));
  std::vector<float> rows(e.output_elements);
  check(cudaMemcpyAsync(e.input, prepared.blob.data(), e.input_elements * sizeof(float), cudaMemcpyHostToDevice,
                        e.stream),
        "copy to GPU");
  if (!e.context->enqueueV3(e.stream)) throw std::runtime_error("TensorRT failed to run the engine");
  check(cudaMemcpyAsync(rows.data(), e.output, e.output_elements * sizeof(float), cudaMemcpyDeviceToHost, e.stream),
        "copy from GPU");
  check(cudaStreamSynchronize(e.stream), "cudaStreamSynchronize");
  return decode(rows, prepared, min_score_);
}

}  // namespace tether
