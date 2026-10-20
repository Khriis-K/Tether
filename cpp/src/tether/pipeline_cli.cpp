#include "tether/pipeline_cli.hpp"

#include <chrono>
#include <csignal>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "tether/detector.hpp"
#include "tether/formats.hpp"
#include "tether/frame_prep.hpp"
#include "tether/pipeline.hpp"
#include "tether/tracker.hpp"
#include "tether/video_source.hpp"

namespace tether {

namespace {

volatile std::sig_atomic_t interrupted = 0;

void on_interrupt(int) { interrupted = 1; }

// Ctrl-C asks the run to stop after the current frame, for as long as this lives.
class InterruptGuard {
 public:
  InterruptGuard() : previous_(std::signal(SIGINT, on_interrupt)) { interrupted = 0; }
  ~InterruptGuard() {
    if (previous_ != SIG_ERR) std::signal(SIGINT, previous_);
  }

 private:
  void (*previous_)(int);
};

struct Options {
  std::string source;
  std::optional<std::string> model, dump, timing;
  std::string model_size = "n";
  double det_min_score = DETECTOR_MIN_SCORE;
  long long max_frames = 0;  // 0: no limit
};

struct Option {
  std::string metavar, help;
  std::function<void(const std::string&)> set;  // throws std::invalid_argument on a bad value
};

std::string usage() { return "usage: tether-pipeline [options] source\n"; }

long long parse_positive(const std::string& text) {
  size_t used = 0;
  long long v = std::stoll(text, &used);
  if (used != text.size() || v < 1) throw std::invalid_argument(text);
  return v;
}

double parse_double(const std::string& text) {
  size_t used = 0;
  double v = std::stod(text, &used);
  if (used != text.size()) throw std::invalid_argument(text);
  return v;
}

std::string default_timing_log() {
  auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm local{};
#ifdef _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  std::ostringstream name;
  name << "results/timing-cpp-" << std::put_time(&local, "%Y%m%d-%H%M%S") << ".jsonl";
  return name.str();
}

}  // namespace

std::filesystem::path model_for_size(const std::string& size) {
  if (size != "n" && size != "s" && size != "m") throw std::invalid_argument("model size must be n, s or m");
  return std::filesystem::path("models") / ("yolo26" + size + ".onnx");
}

int run_pipeline_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  Options o;
  std::map<std::string, Option> options = {
      {"--model", {"MODEL", "YOLO26 ONNX model; overrides --model-size (see tools/export_model.py)",
                   [&](const std::string& v) { o.model = v; }}},
      {"--model-size", {"SIZE", "n, s or m: runs models/yolo26<SIZE>.onnx (default n)",
                        [&](const std::string& v) {
                          model_for_size(v);  // throws on anything but n, s or m
                          o.model_size = v;
                        }}},
      {"--dump", {"DETECTIONS", "write a detection file (.jsonl) instead of tracking",
                  [&](const std::string& v) { o.dump = v; }}},
      {"--det-min-score", {"SCORE", "lowest detector score kept; the tracker then applies its own threshold "
                                    "on top (default 0.1)",
                           [&](const std::string& v) { o.det_min_score = parse_double(v); }}},
      {"--timing", {"LOG", "timing log to write (.jsonl; default results/timing-cpp-<date>-<time>.jsonl)",
                    [&](const std::string& v) { o.timing = v; }}},
      {"--max-frames", {"N", "stop after N frames (default: run to the end, or until Ctrl-C)",
                        [&](const std::string& v) { o.max_frames = parse_positive(v); }}},
  };

  std::vector<std::string> positional;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      out << usage() << "\nRun YOLO26 + the tracker on a webcam or video file, headless, logging each frame's "
          << "step times.\nWith --dump, skip the tracker and write the raw detections instead.\n\n"
          << "positional arguments:\n  source  webcam index (e.g. \"0\") or video file path\n\noptions:\n";
      for (const auto& [name, option] : options)
        out << "  " << name << " " << option.metavar << "\n      " << option.help << "\n";
      return 0;
    }
    if (arg.empty() || arg[0] != '-') {
      positional.push_back(arg);
      continue;
    }
    auto eq = arg.find('=');
    auto name = arg.substr(0, eq);
    auto it = options.find(name);
    if (it == options.end()) {
      err << usage() << "tether-pipeline: error: unrecognized argument " << arg << "\n";
      return 2;
    }
    std::string value;
    if (eq != std::string::npos) {
      value = arg.substr(eq + 1);
    } else if (i + 1 < args.size()) {
      value = args[++i];
    } else {
      err << usage() << "tether-pipeline: error: " << name << " needs a value\n";
      return 2;
    }
    try {
      it->second.set(value);
    } catch (const std::exception&) {  // invalid_argument or out_of_range
      err << usage() << "tether-pipeline: error: " << name << ": invalid value '" << value << "'\n";
      return 2;
    }
  }
  if (positional.size() != 1) {
    err << usage() << "tether-pipeline: error: expected one source\n";
    return 2;
  }
  o.source = positional[0];

  try {
    OnnxDetector detector(o.model ? std::filesystem::path(*o.model) : model_for_size(o.model_size), o.det_min_score);
    auto [height, width] = detector.input_size();
    CpuFramePrep prep(height, width);
    auto source = open_video_source(o.source);
    InterruptGuard guard;
    long long frames = 0;
    auto more = [&] { return !interrupted && (o.max_frames == 0 || frames < o.max_frames); };

    auto make_parent = [](const std::filesystem::path& p) {
      if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    };
    if (o.dump) {
      make_parent(*o.dump);
      DetectionWriter writer(*o.dump);
      dump_detections(*source, prep, detector, [&](const std::vector<Detection>& d) {
        writer.write(d);
        ++frames;
      }, more);
      out << frames << " frames of detections saved to " << *o.dump << "\n";
      return 0;
    }

    std::filesystem::path log_path = o.timing ? *o.timing : default_timing_log();
    make_parent(log_path);
    TimingLogWriter log(log_path);
    out << "logging timing to " << log_path.string() << "\n";
    Tracker tracker;
    std::optional<int> dropped;  // stays empty when the source can't count drops
    // The draw step, with nothing to draw: it only says whether to go on.
    auto headless = [&](const cv::Mat&, const std::vector<Track>&) {
      ++frames;
      return more();
    };
    run_sequential(*source, prep, detector, tracker, headless, steady_clock_ms, [&](const StepTimes& t) {
      log.write(t);
      if (t.dropped) dropped = dropped.value_or(0) + *t.dropped;
    });
    out << frames << " frames";
    if (dropped) out << ", " << *dropped << " dropped by the camera";
    out << "\n";
  } catch (const std::exception& e) {
    err << "tether-pipeline: error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace tether
