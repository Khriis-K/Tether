// tether-pipeline: run YOLO26 + the tracker on a webcam or video file, headless
// unless asked to show the tracks in a window (--show) or record them (--record).
// The C++ counterpart of tether-demo (python/tether/demo.py).
#pragma once

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include "tether/detector.hpp"
#include "tether/frame_prep.hpp"
#include "tether/panel.hpp"
#include "tether/pipeline.hpp"

namespace tether {

// models/yolo26<size>.onnx for size "n", "s" or "m"; throws std::invalid_argument otherwise.
std::filesystem::path model_for_size(const std::string& size);

// The speed settings behind a run of `model` in `mode` on `runtime` (TensorRT
// with a CUDA graph unless `cuda_graph` is false) with frames prepared by
// `prep`, for the stats panel. Only the non-default graph setting is named.
Settings run_settings(const std::filesystem::path& model, PipelineMode mode = PipelineMode::sequential,
                      Runtime runtime = Runtime::onnxruntime, bool cuda_graph = true, Prep prep = Prep::cpu);

// `args` excludes the program name. Exit codes as tether-track: 0 ok, 2 bad
// arguments, 1 when the source, model or an output file can't be used.
// Ctrl-C ends the run cleanly, leaving valid output files.
int run_pipeline_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace tether
