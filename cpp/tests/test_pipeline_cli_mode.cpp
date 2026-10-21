// tether-pipeline --pipeline: sequential or parallel, through the stand-in model
// (tests/onnx/make_models.py).

#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;
using tether::test::temp_path;

namespace {

const std::string MODEL = std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx";

// `n` dark grey 1920x1080 frames as an image sequence; returns its pattern.
std::string frames(int n) {
  auto dir = temp_path("frames").parent_path();
  for (int i = 0; i < n; ++i)
    cv::imwrite((dir / ("f" + std::to_string(i) + ".png")).string(), cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(20, 20, 20)));
  return (dir / "f%d.png").string();
}

int cli(std::vector<std::string> args, std::string* err_text = nullptr, std::string* out_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  if (out_text) *out_text = out.str();
  return code;
}

std::vector<nlohmann::json> json_lines(const std::filesystem::path& path) {
  std::ifstream f(path);
  std::vector<nlohmann::json> lines;
  for (std::string line; std::getline(f, line);) lines.push_back(nlohmann::json::parse(line));
  return lines;
}

std::vector<cv::Mat> read_video(const std::filesystem::path& path) {
  cv::VideoCapture cap(path.string());
  std::vector<cv::Mat> out;
  for (cv::Mat f; cap.read(f);) out.push_back(f.clone());
  return out;
}

}  // namespace

TEST(RunSettingsMode, ThePipelineRowNamesTheModeAndItsThreads) {
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::sequential).at(3),
            (std::pair<std::string, std::string>{"pipeline", "sequential, 1 thread"}));
  EXPECT_EQ(run_settings("models/yolo26n.onnx", PipelineMode::parallel).at(3),
            (std::pair<std::string, std::string>{"pipeline", "parallel, 5 threads"}));
}

TEST(PipelineCliMode, ParallelRecordsTheSameTracksAsSequential) {
  auto source = frames(8);
  auto sequential = temp_path("sequential.avi"), parallel = temp_path("parallel.avi");
  ASSERT_EQ(cli({source, "--model", MODEL, "--timing", temp_path("s.jsonl").string(), "--record",
                 sequential.string()}),
            0);
  ASSERT_EQ(cli({source, "--model", MODEL, "--timing", temp_path("p.jsonl").string(), "--record",
                 parallel.string(), "--pipeline", "parallel"}),
            0);

  auto a = read_video(sequential), b = read_video(parallel);
  ASSERT_EQ(a.size(), 8u);
  ASSERT_EQ(b.size(), 8u);
  for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(cv::norm(a[i], b[i], cv::NORM_INF), 0) << "frame " << i;
}

TEST(PipelineCliMode, ParallelLogsEveryFrameWithItsStepsInOrder) {
  auto log = temp_path("p.jsonl");
  std::string out;
  ASSERT_EQ(cli({frames(6), "--model", MODEL, "--timing", log.string(), "--pipeline", "parallel"}, nullptr, &out), 0);

  auto lines = json_lines(log);
  ASSERT_EQ(lines.size(), 6u);
  for (const auto& line : lines) {
    double last_end = -1e300;
    for (const char* step : {"capture", "prep", "detect", "track", "draw"}) {
      double start = line["steps"][step][0], end = line["steps"][step][1];
      EXPECT_LE(last_end, start) << step;
      EXPECT_LE(start, end) << step;
      last_end = end;
    }
  }
  EXPECT_NE(out.find("6 frames"), std::string::npos) << out;
}

// The tracks are the same either way, so the log is how to tell the modes
// apart: only in parallel does a frame's capture start before the frame
// ahead of it is drawn.
TEST(PipelineCliMode, OnlyParallelOverlapsFramesInTheLog) {
  auto source = frames(6);
  auto overlapped = [&](const std::string& mode) {
    auto log = temp_path(mode + ".jsonl");
    EXPECT_EQ(cli({source, "--model", MODEL, "--timing", log.string(), "--pipeline", mode}), 0);
    auto lines = json_lines(log);
    int count = 0;
    for (size_t i = 1; i < lines.size(); ++i)
      count += double(lines[i]["steps"]["capture"][0]) < double(lines[i - 1]["steps"]["draw"][1]);
    return count;
  };
  EXPECT_EQ(overlapped("sequential"), 0);
  EXPECT_GT(overlapped("parallel"), 0);
}

TEST(PipelineCliMode, MaxFramesStopsParallelAtExactlyThatManyFrames) {
  auto log = temp_path("p.jsonl");
  ASSERT_EQ(cli({frames(10), "--model", MODEL, "--timing", log.string(), "--pipeline", "parallel", "--max-frames",
                 "3"}),
            0);
  EXPECT_EQ(json_lines(log).size(), 3u);  // frames read ahead aren't logged
}

TEST(PipelineCliMode, SequentialIsTheDefaultAndCanBeAskedFor) {
  auto log = temp_path("s.jsonl");
  ASSERT_EQ(cli({frames(2), "--model", MODEL, "--timing", log.string(), "--pipeline", "sequential"}), 0);
  EXPECT_EQ(json_lines(log).size(), 2u);
}

TEST(PipelineCliMode, AnUnknownModeIsABadArgument) {
  std::string err;
  EXPECT_EQ(cli({"a.mp4", "--pipeline", "fast"}, &err), 2);
  EXPECT_NE(err.find("--pipeline"), std::string::npos) << err;
}

TEST(PipelineCliMode, ParallelDoesntGoWithDump) {
  std::string err;
  EXPECT_EQ(cli({frames(1), "--model", MODEL, "--dump", temp_path("d.jsonl").string(), "--pipeline", "parallel"},
                &err),
            2);
  EXPECT_NE(err.find("--dump"), std::string::npos) << err;
}
