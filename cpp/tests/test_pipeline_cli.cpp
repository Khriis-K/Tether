// tether-pipeline end to end: an image sequence through the stand-in model
// (tests/onnx/make_models.py), and argument handling.

#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;
using tether::test::boxes_near;
using tether::test::temp_path;

namespace {

const std::string MODEL = std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx";

// `n` black 1920x1080 frames as an image sequence; returns its pattern.
std::string frames(int n) {
  auto dir = temp_path("frames").parent_path();
  for (int i = 0; i < n; ++i)
    cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0)));
  return (dir / "f%d.png").string();
}

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

std::vector<nlohmann::json> json_lines(const std::filesystem::path& path) {
  std::ifstream f(path);
  std::vector<nlohmann::json> lines;
  for (std::string line; std::getline(f, line);) lines.push_back(nlohmann::json::parse(line));
  return lines;
}

}  // namespace

TEST(ModelForSize, PicksTheYolo26ModelOfThatSize) {
  EXPECT_EQ(model_for_size("n"), std::filesystem::path("models/yolo26n.onnx"));
  EXPECT_EQ(model_for_size("s"), std::filesystem::path("models/yolo26s.onnx"));
  EXPECT_EQ(model_for_size("m"), std::filesystem::path("models/yolo26m.onnx"));
  EXPECT_THROW(model_for_size("x"), std::invalid_argument);
}

TEST(PipelineCli, DumpWritesEveryFramesDetections) {
  auto dets = temp_path("dets.jsonl");

  ASSERT_EQ(cli({frames(3), "--model", MODEL, "--dump", dets.string()}), 0);

  auto loaded = load_detections(dets);
  ASSERT_EQ(loaded.size(), 3u);
  for (const auto& frame : loaded) {
    ASSERT_EQ(frame.size(), 1u);
    EXPECT_TRUE(boxes_near(frame[0].box, {150.0, 150.0, 450.0, 450.0}, 1e-3));
  }
}

TEST(PipelineCli, TrackingWritesATimingLineForEveryFrame) {
  auto log = temp_path("timing.jsonl");

  ASSERT_EQ(cli({frames(3), "--model", MODEL, "--timing", log.string()}), 0);

  auto lines = json_lines(log);
  ASSERT_EQ(lines.size(), 3u);
  for (size_t i = 0; i < lines.size(); ++i) {
    EXPECT_EQ(lines[i]["frame"], i);
    EXPECT_FALSE(lines[i].contains("dropped"));  // a file can't drop frames
    double last = lines[i]["steps"]["capture"][0];
    for (const char* step : {"capture", "prep", "detect", "track", "draw"}) {  // one after another
      auto span = lines[i]["steps"][step];
      EXPECT_LE(last, span[0].get<double>()) << step;
      EXPECT_LE(span[0].get<double>(), span[1].get<double>()) << step;
      last = span[1];
    }
  }
}

TEST(PipelineCli, MaxFramesStopsTheRunEarly) {
  auto log = temp_path("timing.jsonl");

  ASSERT_EQ(cli({frames(3), "--model", MODEL, "--timing", log.string(), "--max-frames", "2"}), 0);

  EXPECT_EQ(json_lines(log).size(), 2u);
}

TEST(PipelineCli, AnUnknownModelSizeIsABadArgument) {
  std::string err;
  EXPECT_EQ(cli({frames(1), "--model-size", "x"}, &err), 2);
  EXPECT_NE(err.find("--model-size"), std::string::npos) << err;
}

TEST(PipelineCli, BadArgumentsExitWithTwo) {
  EXPECT_EQ(cli({}), 2);  // no source
  EXPECT_EQ(cli({"a.mp4", "b.mp4"}), 2);
  EXPECT_EQ(cli({"a.mp4", "--frobnicate", "1"}), 2);
  EXPECT_EQ(cli({"a.mp4", "--max-frames", "zero"}), 2);
  EXPECT_EQ(cli({"a.mp4", "--max-frames", "0"}), 2);
  EXPECT_EQ(cli({"a.mp4", "--timing"}), 2);  // needs a value
}

TEST(PipelineCli, AMissingSourceOrModelExitsWithOne) {
  std::string err;
  EXPECT_EQ(cli({temp_path("missing.mp4").string(), "--model", MODEL}, &err), 1);
  EXPECT_NE(err.find("missing.mp4"), std::string::npos) << err;
  EXPECT_EQ(cli({frames(1), "--model", temp_path("missing.onnx").string()}), 1);
}

TEST(PipelineCli, HelpExitsWithZero) { EXPECT_EQ(cli({"--help"}), 0); }
