// tether-pipeline: run YOLO26 + the tracker on a webcam or video file, headless
// unless asked to show the tracks in a window (--show) or record them (--record).
// The C++ counterpart of tether-demo (python/tether/demo.py).
#pragma once

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include "tether/panel.hpp"
#include "tether/pipeline.hpp"

namespace tether {

// models/yolo26<size>.onnx for size "n", "s" or "m"; throws std::invalid_argument otherwise.
std::filesystem::path model_for_size(const std::string& size);

// The speed settings behind a run of `model` in `mode`, for the stats panel.
Settings run_settings(const std::filesystem::path& model, PipelineMode mode = PipelineMode::sequential);

// `args` excludes the program name. Exit codes as tether-track: 0 ok, 2 bad
// arguments, 1 when the source, model or an output file can't be used.
// Ctrl-C ends the run cleanly, leaving valid output files.
int run_pipeline_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace tether
