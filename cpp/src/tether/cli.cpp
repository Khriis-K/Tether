#include "tether/cli.hpp"

#include <exception>
#include <functional>
#include <map>
#include <stdexcept>

#include "tether/formats.hpp"
#include "tether/tracker.hpp"

namespace tether {

namespace {

struct Option {
  std::string help;
  std::function<void(const std::string&)> set;  // throws std::invalid_argument on a bad value
};

int parse_int(const std::string& text) {
  size_t used = 0;
  int v = std::stoi(text, &used);
  if (used != text.size()) throw std::invalid_argument(text);
  return v;
}

double parse_double(const std::string& text) {
  size_t used = 0;
  double v = std::stod(text, &used);  // also reads "inf", like Python's float()
  if (used != text.size()) throw std::invalid_argument(text);
  return v;
}

std::string usage() {
  return "usage: tether-track [options] detections tracks\n";
}

}  // namespace

int run_track_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  TrackerSettings s;
  auto num = [](double v) {
    std::string text = std::to_string(v);
    text.erase(text.find_last_not_of('0') + 1);
    if (text.back() == '.') text += '0';
    return text;
  };
  std::map<std::string, Option> options = {
      {"--confirm-frames",
       {"matched frames in a row before a new track is confirmed (default " + std::to_string(CONFIRM_FRAMES) + ")",
        [&](const std::string& v) { s.confirm_frames = parse_int(v); }}},
      {"--max-missing-frames",
       {"most missed frames in a row a searching track survives; it is removed on the next miss (default " +
            std::to_string(MAX_MISSING_FRAMES) + ")",
        [&](const std::string& v) { s.max_missing_frames = parse_int(v); }}},
      {"--min-score",
       {"detections scored below this are ignored (default " + num(MIN_SCORE) + ")",
        [&](const std::string& v) { s.min_score = parse_double(v); }}},
      {"--search-gate",
       {"center-distance gate, in track box sizes, for re-matching a searching track; this and "
        "--search-gate-growth both 0 turns it off (default " +
            num(SEARCH_GATE) + ")",
        [&](const std::string& v) { s.search_gate = parse_double(v); }}},
      {"--search-gate-growth",
       {"added to the gate per frame the track has missed (default " + num(SEARCH_GATE_GROWTH) + ")",
        [&](const std::string& v) { s.search_gate_growth = parse_double(v); }}},
      {"--search-gate-cap",
       {"caps the gate at this times the distance to the nearest other confirmed track; inf is no cap (default " +
            num(SEARCH_GATE_CAP) + ")",
        [&](const std::string& v) { s.search_gate_cap = parse_double(v); }}},
      {"--coast-damping",
       {"velocity is multiplied by this each frame a track goes unmatched; 1 is undamped, 0 holds the position "
        "still (default " +
            num(COAST_DAMPING) + ")",
        [&](const std::string& v) { s.coast_damping = parse_double(v); }}},
      {"--merge-reach",
       {"how far, in box sizes, a newly confirmed track can be from an older track's prediction and still be "
        "merged into it as a duplicate; this and --merge-reach-growth both 0 turns merging off (default " +
            num(MERGE_REACH) + ")",
        [&](const std::string& v) { s.merge_reach = parse_double(v); }}},
      {"--merge-reach-growth",
       {"added to the merge reach per frame the older track has missed (default " + num(MERGE_REACH_GROWTH) + ")",
        [&](const std::string& v) { s.merge_reach_growth = parse_double(v); }}},
  };

  std::vector<std::string> paths;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg == "-h" || arg == "--help") {
      out << usage() << "\nRun the tracker on a detection file and write a track file.\n\n"
          << "positional arguments:\n  detections  input detection file (.jsonl)\n"
          << "  tracks      output track file (.jsonl)\n\noptions:\n";
      for (const auto& [name, option] : options) out << "  " << name << " VALUE\n      " << option.help << "\n";
      return 0;
    }
    if (arg.rfind("--", 0) != 0) {
      paths.push_back(arg);
      continue;
    }
    auto it = options.find(arg);
    if (it == options.end()) {
      err << usage() << "tether-track: error: unrecognized argument " << arg << "\n";
      return 2;
    }
    if (i + 1 >= args.size()) {
      err << usage() << "tether-track: error: " << arg << " needs a value\n";
      return 2;
    }
    try {
      it->second.set(args[++i]);
    } catch (const std::exception&) {  // invalid_argument or out_of_range
      err << usage() << "tether-track: error: " << arg << ": invalid value '" << args[i] << "'\n";
      return 2;
    }
  }
  if (paths.size() != 2) {
    err << usage() << "tether-track: error: expected a detection file and a track file\n";
    return 2;
  }

  try {
    Tracker tracker(s);
    TrackFrames frames;
    for (const auto& dets : load_detections(paths[0])) frames.push_back(tracker.update(dets));
    save_tracks(paths[1], frames);
  } catch (const std::exception& e) {
    err << "tether-track: error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace tether
