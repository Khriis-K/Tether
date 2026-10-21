// tether-time-track: times the tracking step alone, for the Python-vs-C++ speed check (#10).

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/timing.hpp"

using namespace tether;
using namespace tether::test;

namespace {

// One object walking right for `n` frames.
DetectionFrames walk(int n) {
  DetectionFrames frames;
  for (int f = 0; f < n; ++f) frames.push_back({det({5.0 * f, 0.0, 5.0 * f + 20.0, 20.0})});
  return frames;
}

int time_cli(std::vector<std::string> args, std::string* out_text = nullptr, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_time_cli(args, out, err);
  if (out_text) *out_text = out.str();
  if (err_text) *err_text = err.str();
  return code;
}

}  // namespace

TEST(TimeTracking, GivesOneTimePerRepeat) {
  auto times = time_tracking(walk(50), 4);

  ASSERT_EQ(times.size(), 4u);
  for (double ms : times) EXPECT_GE(ms, 0.0);
}

TEST(TimeTracking, ALongerInputTakesTime) {
  // 5,000 tracker updates can't take under 10 ns each (0.05 ms in all); a time
  // that small means the clock didn't surround the loop, or the loop was optimized away.
  auto times = time_tracking(walk(5000), 1);

  EXPECT_GT(times[0], 0.05);
}

TEST(TimeCli, PrintsFrameCountAndOneTimePerRepeat) {
  auto path = temp_path("dets.jsonl");
  save_detections(path, walk(30));

  std::string out;
  ASSERT_EQ(time_cli({path.string(), "--repeats", "3"}, &out), 0);

  auto result = nlohmann::json::parse(out);
  EXPECT_EQ(result["frames"], 30);
  ASSERT_EQ(result["repeats_ms"].size(), 3u);
  for (const auto& ms : result["repeats_ms"]) EXPECT_GE(ms.get<double>(), 0.0);
}

TEST(TimeCli, RepeatsDefaultToFive) {
  auto path = temp_path("dets.jsonl");
  save_detections(path, walk(3));

  std::string out;
  ASSERT_EQ(time_cli({path.string()}, &out), 0);

  EXPECT_EQ(nlohmann::json::parse(out)["repeats_ms"].size(), 5u);
}

TEST(TimeCli, AcceptsRepeatsWithEquals) {
  auto path = temp_path("dets.jsonl");
  save_detections(path, walk(3));

  std::string out;
  ASSERT_EQ(time_cli({path.string(), "--repeats=2"}, &out), 0);

  EXPECT_EQ(nlohmann::json::parse(out)["repeats_ms"].size(), 2u);
}

TEST(TimeCli, RejectsBadArguments) {
  auto path = temp_path("dets.jsonl").string();
  save_detections(path, walk(3));

  std::string err;
  EXPECT_EQ(time_cli({}, nullptr, &err), 2);
  EXPECT_NE(err.find("usage"), std::string::npos);
  EXPECT_EQ(time_cli({path, "--repeats", "0"}), 2);
  EXPECT_EQ(time_cli({path, "--repeats", "two"}), 2);
  EXPECT_EQ(time_cli({path, "--repeats", "2x"}), 2);
  EXPECT_EQ(time_cli({path, "--repeats"}), 2);
  EXPECT_EQ(time_cli({path, "--frobnicate", "1"}), 2);
  EXPECT_EQ(time_cli({path, path}), 2);
}

TEST(TimeCli, MissingFileIsAnError) {
  std::string err;
  EXPECT_EQ(time_cli({temp_path("nope.jsonl").string()}, nullptr, &err), 1);
  EXPECT_NE(err.find("tether-time-track: error"), std::string::npos);
}

TEST(TimeCli, HelpExitsZero) {
  std::string out;
  EXPECT_EQ(time_cli({"--help"}, &out), 0);
  EXPECT_NE(out.find("--repeats"), std::string::npos);
}
