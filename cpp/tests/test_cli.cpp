// tether-track: detection file in, track file out, with the Python CLI's options.
// Ported from the CLI tests in python/tests (test_cli.py and the
// test_cli_passes_* tests of each tracker behavior file).

#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/cli.hpp"
#include "tether/formats.hpp"

using namespace tether;
using namespace tether::test;

namespace {

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_track_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

// Writes `frames` to a detection file, runs the CLI with `options`, reads the track file back.
TrackFrames track(const DetectionFrames& frames, const std::vector<std::string>& options, const std::string& name) {
  auto dets_path = temp_path(name + ".dets.jsonl");
  auto tracks_path = temp_path(name + ".tracks.jsonl");
  save_detections(dets_path, frames);
  std::vector<std::string> args = {dets_path.string(), tracks_path.string()};
  args.insert(args.end(), options.begin(), options.end());
  EXPECT_EQ(cli(args), 0);
  return load_tracks(tracks_path);
}

Box box_at(double x, double y = 100.0, double size = 20.0) { return {x, y, x + size, y + size}; }

// Two objects moving steadily, plus jittered noise boxes, for a busy input.
DetectionFrames scripted_scene(int n_frames = 40, unsigned seed = 0) {
  std::mt19937 rng(seed);
  auto uniform = [&](double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); };
  DetectionFrames frames;
  for (int f = 0; f < n_frames; ++f) {
    std::vector<Detection> dets = {
        {{2.0 * f, 50.0, 2.0 * f + 20.0, 70.0}, 0.9, 0},
        {{300.0 - 3.0 * f, 200.0, 330.0 - 3.0 * f, 240.0}, 0.8, 2},
    };
    int noise = std::uniform_int_distribution<int>(0, 3)(rng);
    for (int i = 0; i < noise; ++i) {
      double x = uniform(0, 600), y = uniform(0, 400);
      dets.push_back({{x, y, x + uniform(5, 40), y + uniform(5, 40)}, uniform(0.1, 0.6),
                      std::uniform_int_distribution<int>(0, 2)(rng)});
    }
    std::shuffle(dets.begin(), dets.end(), rng);
    frames.push_back(dets);
  }
  return frames;
}

}  // namespace

TEST(Cli, WritesOneTrackFramePerDetectionFrame) {
  auto frames = track({{det({0.0, 0.0, 10.0, 10.0})}, {}}, {}, "one");

  ASSERT_EQ(frames.size(), 2u);
  EXPECT_EQ(frames[0].size(), 1u);
  EXPECT_TRUE(frames[1].empty());
}

TEST(Cli, SameInputRunTwiceGivesByteIdenticalOutput) {
  auto dets_path = temp_path("dets.jsonl");
  save_detections(dets_path, scripted_scene());
  auto out_a = temp_path("a.jsonl"), out_b = temp_path("b.jsonl");

  cli({dets_path.string(), out_a.string()});
  cli({dets_path.string(), out_b.string()});

  EXPECT_EQ(read_bytes(out_a), read_bytes(out_b));
  EXPECT_FALSE(read_bytes(out_a).empty());
}

TEST(Cli, PassesLifecycleSettingsToTheTracker) {
  auto frames = track({{det(box_at(0.0))}, {}, {}, {}}, {"--confirm-frames", "1", "--max-missing-frames", "2"}, "life");

  std::vector<std::vector<std::string>> statuses;
  for (const auto& tracks : frames) {
    statuses.emplace_back();
    for (const auto& t : tracks) statuses.back().push_back(t.status);
  }
  EXPECT_EQ(statuses, (std::vector<std::vector<std::string>>{{"tracked"}, {"searching"}, {"searching"}, {}}));
}

TEST(Cli, PassesTheMinScoreToTheTracker) {
  auto frames = track({{det(box_at(0.0), 0.4)}}, {"--min-score", "0.5"}, "score");

  EXPECT_TRUE(frames[0].empty());
}

TEST(Cli, PassesTheGateSettingsToTheTracker) {
  // A searching track and a detection 1.25 box sizes away, inside a gate of
  // 1 + 0.5 after one missed frame but not overlapping (test_search_gate.cpp).
  DetectionFrames frames(5, {det(box_at(100.0))});
  frames.push_back({});
  frames.push_back({det(box_at(125.0))});

  auto off = track(frames, {"--confirm-frames", "3", "--search-gate", "0", "--search-gate-growth", "0"}, "off");
  auto on = track(frames, {"--confirm-frames", "3", "--search-gate", "1", "--search-gate-growth", "0.5"}, "on");

  EXPECT_NE(off.back().back().id, off[off.size() - 2][0].id);
  EXPECT_EQ(only(on.back()).id, only(on[on.size() - 2]).id);
}

TEST(Cli, PassesTheCapToTheTracker) {
  // As above, with a neighbor 2 box sizes away: a cap of 0.5 shrinks the gate to 1.
  Box neighbor = box_at(60.0);
  DetectionFrames frames(5, {det(box_at(100.0)), det(neighbor)});
  frames.push_back({det(neighbor)});
  frames.push_back({det(box_at(125.0)), det(neighbor)});
  std::vector<std::string> gate = {"--confirm-frames", "3", "--search-gate", "1", "--search-gate-growth", "0.5"};
  auto with = [&](std::string cap) {
    auto opts = gate;
    opts.insert(opts.end(), {"--search-gate-cap", cap});
    return opts;
  };

  auto uncapped = track(frames, with("inf"), "inf");
  auto capped = track(frames, with("0.5"), "cap");

  int a = uncapped[4][0].id;  // A has the lower ID: it was detected first
  auto id_at_return = [](const TrackFrames& out) {
    for (const auto& t : out.back())
      if (t.box == box_at(125.0)) return t.id;
    return -1;
  };
  EXPECT_EQ(id_at_return(uncapped), a);
  EXPECT_NE(id_at_return(capped), a);
}

TEST(Cli, PassesTheMergeReachToTheTracker) {
  // A confirmed, then seen only 6 box sizes away: past the gate, inside the merge reach.
  DetectionFrames frames(5, {det(box_at(400.0))});
  for (int i = 0; i < 3; ++i) frames.push_back({det(box_at(520.0))});

  auto on = track(frames, {"--confirm-frames", "3"}, "on");
  auto off = track(frames, {"--confirm-frames", "3", "--merge-reach", "0", "--merge-reach-growth", "0"}, "off");

  int a = on[4][0].id;
  EXPECT_EQ(only(on.back()).id, a);
  EXPECT_NE(off.back().back().id, a);
}

TEST(Cli, PassesTheDampingToTheTracker) {
  DetectionFrames frames;
  for (int f = 0; f < 10; ++f) frames.push_back({det({10.0 * f, 0.0, 10.0 * f + 20.0, 20.0})});
  for (int i = 0; i < 5; ++i) frames.push_back({});

  auto out = track(frames, {"--confirm-frames", "3", "--coast-damping", "0"}, "damp");

  for (size_t f = 10; f < out.size(); ++f) EXPECT_EQ(only(out[f]).predicted_box, only(out[9]).predicted_box);
}

TEST(Cli, AcceptsOptionEqualsValue) {
  auto frames = track({{det(box_at(0.0), 0.4)}}, {"--min-score=0.5"}, "equals");

  EXPECT_TRUE(frames[0].empty());
}

TEST(Cli, HelpListsTheDefaults) {
  std::ostringstream out, err;
  ASSERT_EQ(run_track_cli({"--help"}, out, err), 0);

  EXPECT_NE(out.str().find("(default 0.25)"), std::string::npos);  // --min-score
  EXPECT_NE(out.str().find("(default 30)"), std::string::npos);    // --max-missing-frames
}

TEST(Cli, HelpExitsZero) { EXPECT_EQ(cli({"--help"}), 0); }

TEST(Cli, BadArgumentsAreAnErrorWithAMessage) {
  // Real files, so the only thing wrong is the argument: exit 2, as argparse.
  auto dets = temp_path("dets.jsonl").string();
  auto out = temp_path("out.jsonl").string();
  save_detections(dets, {{}});
  ASSERT_EQ(cli({dets, out}), 0);
  for (std::vector<std::string> args : {std::vector<std::string>{dets},
                                        {dets, out, "--no-such-option", "1"},
                                        {dets, out, "--confirm-frames"},
                                        {dets, out, "--confirm-frames", "three"},
                                        {dets, out, "--confirm-frames", "3.5"},
                                        {dets, out, "--min-score", "0.5x"},
                                        {dets, out, "extra"},
                                        {dets, out, "-x"},
                                        {dets, out, "--min-score="}}) {
    std::string err;
    EXPECT_EQ(cli(args, &err), 2) << args.back();
    EXPECT_FALSE(err.empty()) << args.back();
  }
}

TEST(Cli, MissingDetectionFileIsAnErrorWithAMessage) {
  std::string err;

  EXPECT_NE(cli({temp_path("missing.jsonl").string(), temp_path("out.jsonl").string()}, &err), 0);
  EXPECT_NE(err.find("missing.jsonl"), std::string::npos);
}
