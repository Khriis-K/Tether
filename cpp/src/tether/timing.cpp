#include "tether/timing.hpp"

#include <chrono>
#include <exception>
#include <nlohmann/json.hpp>

#include "tether/tracker.hpp"

namespace tether {

namespace {

std::string usage() { return "usage: tether-time-track [--repeats N] detections\n"; }

int parse_repeats(const std::string& text) {
  size_t used = 0;
  int v = std::stoi(text, &used);
  if (used != text.size() || v < 1) throw std::invalid_argument(text);
  return v;
}

}  // namespace

std::vector<double> time_tracking(const DetectionFrames& frames, int repeats) {
  std::vector<double> times;
  for (int r = 0; r < repeats; ++r) {
    Tracker tracker;
    auto start = std::chrono::steady_clock::now();
    for (const auto& dets : frames) tracker.update(dets);  // outputs built, as in the CLI, then dropped
    auto end = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  return times;
}

int run_time_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  int repeats = 5;
  std::vector<std::string> paths;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      out << usage() << "\nTime the tracking step alone: a fresh default tracker over every frame, once per repeat.\n"
          << "Prints {\"frames\": N, \"repeats_ms\": [...]}.\n\n"
          << "positional arguments:\n  detections  input detection file (.jsonl)\n\noptions:\n"
          << "  --repeats N\n      times to run the whole file (default 5)\n";
      return 0;
    }
    if (arg.empty() || arg[0] != '-') {
      paths.push_back(arg);
      continue;
    }
    auto eq = arg.find('=');
    auto name = arg.substr(0, eq);
    if (name != "--repeats") {
      err << usage() << "tether-time-track: error: unrecognized argument " << arg << "\n";
      return 2;
    }
    std::string value;
    if (eq != std::string::npos) {
      value = arg.substr(eq + 1);
    } else if (i + 1 < args.size()) {
      value = args[++i];
    } else {
      err << usage() << "tether-time-track: error: --repeats needs a value\n";
      return 2;
    }
    try {
      repeats = parse_repeats(value);
    } catch (const std::exception&) {  // invalid_argument or out_of_range
      err << usage() << "tether-time-track: error: --repeats: invalid value '" << value << "'\n";
      return 2;
    }
  }
  if (paths.size() != 1) {
    err << usage() << "tether-time-track: error: expected one detection file\n";
    return 2;
  }

  try {
    auto frames = load_detections(paths[0]);
    nlohmann::json result = {{"frames", frames.size()}, {"repeats_ms", time_tracking(frames, repeats)}};
    out << result.dump() << "\n";
  } catch (const std::exception& e) {
    err << "tether-time-track: error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace tether
