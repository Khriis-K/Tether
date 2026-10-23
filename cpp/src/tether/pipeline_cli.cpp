#include "tether/pipeline_cli.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <opencv2/videoio.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "tether/class_names.hpp"
#include "tether/detector.hpp"
#include "tether/formats.hpp"
#include "tether/frame_prep.hpp"
#include "tether/gui_screen.hpp"
#include "tether/hud_view.hpp"
#include "tether/live_stats.hpp"
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
  std::optional<std::string> model, class_names, dump, timing, record, screenshot;
  bool show = false;
  PipelineMode pipeline = PipelineMode::sequential;
  Runtime runtime = Runtime::onnxruntime;
  bool cuda_graph = true;
  Prep prep = Prep::cpu;
  std::string model_size = "n";
  double det_min_score = DETECTOR_MIN_SCORE;
  long long max_frames = 0;  // 0: no limit
};

struct Option {
  std::string metavar, help;  // no metavar: a flag, which takes no value
  std::function<void(const std::string&)> set;  // throws std::invalid_argument on a bad value
};

constexpr double DEFAULT_RECORD_FPS = 30;  // when the source can't tell its rate (a webcam may not)

// A video writer for `path`: Motion JPEG for .avi, MPEG-4 otherwise (.mp4).
cv::VideoWriter open_recording(const std::filesystem::path& path, double fps, cv::Size size) {
  auto ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  int fourcc = ext == ".avi" ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                              : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  cv::VideoWriter writer(path.string(), fourcc, fps, size);
  if (!writer.isOpened()) throw std::runtime_error("can't record to '" + path.string() + "'");
  return writer;
}

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
  return std::filesystem::path("models") / ("yolo26" + size + "_" + SHIPPED_INPUT + ".onnx");
}

Settings run_settings(const std::filesystem::path& model, PipelineMode mode, Runtime runtime, bool cuda_graph,
                      Prep prep) {
  // ONNX Runtime's own threads inside detect aren't counted in either mode.
  std::string pipeline = mode == PipelineMode::parallel
                             ? "parallel, " + std::to_string(PARALLEL_STAGE_THREADS + 1) + " threads"  // + main
                             : "sequential, 1 thread";
  return {{"model", model.filename().string()},
          {"runtime", runtime == Runtime::onnxruntime ? "ONNX Runtime (CPU)"
                      : cuda_graph                    ? "TensorRT (GPU)"
                                                      : "TensorRT (GPU, no CUDA graph)"},
          {"frame prep", prep == Prep::cuda ? "CUDA (GPU)" : "CPU"},
          {"pipeline", pipeline}};
}

int run_pipeline_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  Options o;
  std::map<std::string, Option> options = {
      {"--model", {"MODEL", "YOLO26 ONNX model; overrides --model-size (see tools/export_model.py)",
                   [&](const std::string& v) { o.model = v; }}},
      {"--model-size", {"SIZE", std::string("n, s or m: runs models/yolo26<SIZE>_") + SHIPPED_INPUT +
                                    ".onnx (default n, the shipped detector)",
                        [&](const std::string& v) {
                          model_for_size(v);  // throws on anything but n, s or m
                          o.model_size = v;
                        }}},
      {"--class-names", {"NAMES", "the stats panel's names for class IDs 0, 1, ..., comma-separated (e.g. drone); "
                                  "default: the model's ONNX metadata, else COCO's names. Needed for a TensorRT engine "
                                  "with no .onnx beside it",
                         [&](const std::string& v) {
                           parse_names_flag(v);  // throws on an empty name
                           o.class_names = v;
                         }}},
      {"--dump", {"DETECTIONS", "write a detection file (.jsonl) instead of tracking",
                  [&](const std::string& v) { o.dump = v; }}},
      {"--det-min-score", {"SCORE", "lowest detector score kept; the tracker then applies its own threshold "
                                    "on top (default 0.1)",
                           [&](const std::string& v) { o.det_min_score = parse_double(v); }}},
      {"--timing", {"LOG", "timing log to write (.jsonl; default results/timing-cpp-<date>-<time>.jsonl)",
                    [&](const std::string& v) { o.timing = v; }}},
      {"--pipeline", {"MODE", "sequential (default): one step at a time; or parallel: capture, prep, detect and track "
                              "each on their own thread, overlapping on different frames, with draw on the main "
                              "thread. Same tracks either way",
                      [&](const std::string& v) {
                        if (v == "sequential") o.pipeline = PipelineMode::sequential;
                        else if (v == "parallel") o.pipeline = PipelineMode::parallel;
                        else throw std::invalid_argument(v);
                      }}},
      {"--runtime", {"RUNTIME", "onnxruntime (default): the model on ONNX Runtime's CPU provider; or tensorrt: "
                                "the TensorRT engine built from it, MODEL with .engine in place of .onnx, on the "
                                "GPU (see README.md, \"TensorRT runtime\")",
                     [&](const std::string& v) {
                       if (v == "onnxruntime") o.runtime = Runtime::onnxruntime;
                       else if (v == "tensorrt") o.runtime = Runtime::tensorrt;
                       else throw std::invalid_argument(v);
                     }}},
      {"--no-cuda-graph", {"", "with --runtime tensorrt: launch the engine's kernels one by one each frame, instead of "
                               "replaying them from a CUDA graph captured once at start-up",
                           [&](const std::string&) { o.cuda_graph = false; }}},
      {"--prep", {"PREP", "cpu (default): prepare frames on the CPU; or cuda: with --runtime tensorrt, on the GPU with "
                          "a CUDA kernel, the prepared frame left there for the engine",
                  [&](const std::string& v) {
                    if (v == "cpu") o.prep = Prep::cpu;
                    else if (v == "cuda") o.prep = Prep::cuda;
                    else throw std::invalid_argument(v);
                  }}},
      {"--max-frames", {"N", "stop after N frames (default: run to the end, or until Ctrl-C)",
                        [&](const std::string& v) { o.max_frames = parse_positive(v); }}},
      {"--show", {"", "show the tracks in a window, with a live stats panel beside the video: Space pauses and resumes, q or Esc quits. While paused "
                      "the pipeline waits, so that frame's draw time in the timing log includes the pause, "
                      "and on a webcam the frames missed meanwhile count as dropped. Time runs unpaused",
                  [&](const std::string&) { o.show = true; }}},
      {"--record", {"VIDEO", "save the video with the tracks drawn (.mp4, or .avi for Motion JPEG), at the "
                             "source's frame rate (30 when it can't tell). One frame per frame tracked, so a webcam "
                             "run that falls behind the camera plays back fast",
                    [&](const std::string& v) { o.record = v; }}},
      {"--screenshot", {"PNG", "with --show: save the window, video and stats panel, as it looks when the run ends",
                        [&](const std::string& v) { o.screenshot = v; }}},
  };

  std::vector<std::string> positional;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      out << usage() << "\nRun YOLO26 + the tracker on a webcam or video file, headless, logging each frame's "
          << "step times.\nWith --dump, skip the tracker and write the raw detections instead.\n\n"
          << "positional arguments:\n  source  webcam index (e.g. \"0\") or video file path\n\noptions:\n";
      for (const auto& [name, option] : options)
        out << "  " << name << (option.metavar.empty() ? "" : " " + option.metavar) << "\n      " << option.help
            << "\n";
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
    if (it->second.metavar.empty()) {
      if (eq != std::string::npos) {
        err << usage() << "tether-pipeline: error: " << name << " takes no value\n";
        return 2;
      }
    } else if (eq != std::string::npos) {
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
  if (o.screenshot && !o.show) {
    err << usage() << "tether-pipeline: error: --screenshot saves the --show window, so it needs --show\n";
    return 2;
  }
  if (o.dump && (o.show || o.record)) {
    err << usage() << "tether-pipeline: error: --show and --record track and draw, which --dump skips\n";
    return 2;
  }
  if (!o.cuda_graph && o.runtime != Runtime::tensorrt) {
    err << usage() << "tether-pipeline: error: --no-cuda-graph only applies to --runtime tensorrt\n";
    return 2;
  }
  if (o.prep == Prep::cuda && o.runtime != Runtime::tensorrt) {
    err << usage() << "tether-pipeline: error: --prep cuda leaves frames on the GPU, so it needs --runtime tensorrt\n";
    return 2;
  }
  if (o.dump && o.pipeline == PipelineMode::parallel) {
    err << usage() << "tether-pipeline: error: --pipeline parallel runs the tracking stages, which --dump skips\n";
    return 2;
  }

  try {
    std::filesystem::path model = o.model ? std::filesystem::path(*o.model) : model_for_size(o.model_size);
    auto detector_ptr = make_detector(o.runtime, model, o.det_min_score, o.cuda_graph);
    Detector& detector = *detector_ptr;
    auto [height, width] = detector.input_size();
    auto prep_ptr = make_frame_prep(o.prep, height, width);
    FramePrep& prep = *prep_ptr;
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

    // The draw step. Headless (neither --show nor --record) it only says whether to go on.
    // The window's stats panel reads the step times the timing log gets, so it
    // shows the frames up to the one before: a frame's own draw time isn't known
    // until it's on screen.
    LiveStats stats;
    std::vector<Track> on_screen;
    std::unique_ptr<GuiScreen> screen;
    if (o.show) {
      ClassNames names = class_names_for(model, o.class_names);
      if (!o.class_names && (model.extension() != ".onnx" || !std::filesystem::exists(model)))
        err << "tether-pipeline: " << model.string() << " isn't an .onnx file to read class names from, so the "
            << "panel uses COCO's; "
            << "pass --class-names to name them\n";
      screen = std::make_unique<GuiScreen>("Tether", [&, names] {
        return PanelData{stats.summary(), on_screen, run_settings(model, o.pipeline, o.runtime, o.cuda_graph, o.prep),
                         names};
      });
    }
    cv::VideoWriter recording;  // opened on the first frame, once its size is known
    Recorder recorder;
    if (o.record) {
      make_parent(*o.record);
      double fps = source->fps() > 0 ? source->fps() : DEFAULT_RECORD_FPS;
      recorder = [&, fps](const cv::Mat& f) {
        if (!recording.isOpened()) recording = open_recording(*o.record, fps, f.size());
        recording.write(f);
      };
    }
    std::optional<HudView> view;
    if (screen || recorder) view.emplace(screen.get(), recorder);
    auto draw = [&](const cv::Mat& frame, const std::vector<Track>& tracks) {
      ++frames;
      if (screen) on_screen = tracks;
      bool keep_going = view ? view->show(frame, tracks) : true;
      return keep_going && more();
    };
    auto run = o.pipeline == PipelineMode::parallel ? run_parallel : run_sequential;
    run(*source, prep, detector, tracker, draw, steady_clock_ms, [&](const StepTimes& t) {
      log.write(t);
      stats.add(t);
      if (t.dropped) dropped = dropped.value_or(0) + *t.dropped;
    });
    out << frames << " frames";
    if (dropped) out << ", " << *dropped << " dropped by the camera";
    out << "\n";
    if (o.screenshot) {
      make_parent(*o.screenshot);
      screen->save_screenshot(*o.screenshot);
      out << "screenshot saved to " << *o.screenshot << "\n";
    }
    if (recording.isOpened()) {
      recording.release();  // finishes the file
      out << "recording saved to " << *o.record << "\n";
    }
  } catch (const std::exception& e) {
    err << "tether-pipeline: error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace tether
