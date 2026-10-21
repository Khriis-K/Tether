// tether-pipeline --record: the drawn run saved to a video, through the
// stand-in model (tests/onnx/make_models.py). --show opens a window, so it's
// checked by running the app, not here.

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/hud.hpp"
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

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

std::vector<cv::Mat> read_video(const std::filesystem::path& path, double* fps = nullptr) {
  cv::VideoCapture cap(path.string());
  if (fps) *fps = cap.get(cv::CAP_PROP_FPS);
  std::vector<cv::Mat> out;
  for (cv::Mat f; cap.read(f);) out.push_back(f.clone());
  return out;
}

int brightness(const cv::Mat& f, int x, int y) {
  auto p = f.at<cv::Vec3b>(y, x);
  return std::max({p[0], p[1], p[2]});
}

}  // namespace

TEST(PipelineCliRecord, SavesEveryFrameWithTheHudDrawn) {
  for (const char* name : {"run.avi", "run.mp4"}) {
    auto video = temp_path(name);

    ASSERT_EQ(cli({frames(5), "--model", MODEL, "--timing", temp_path("t.jsonl").string(), "--record",
                   video.string()}),
              0)
        << name;

    auto recorded = read_video(video);
    ASSERT_EQ(recorded.size(), 5u) << name;
    EXPECT_EQ(recorded[0].size(), cv::Size(1920, 1080)) << name;
    // The stand-in sees one object at (150, 150)-(450, 450). It's confirmed on
    // the third frame, so only then does it get its brackets.
    EXPECT_LT(brightness(recorded[0], 151, 151), 60) << name;
    EXPECT_GT(brightness(recorded[4], 151, 151), 150) << name;
  }
}

TEST(PipelineCliRecord, KeepsTheSourcesFrameRate) {
  auto source = temp_path("source.avi");
  {
    cv::VideoWriter w(source.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 12, {640, 360});
    for (int i = 0; i < 3; ++i) w.write(cv::Mat(360, 640, CV_8UC3, cv::Scalar(20, 20, 20)));
  }
  auto video = temp_path("run.avi");

  ASSERT_EQ(cli({source.string(), "--model", MODEL, "--timing", temp_path("t.jsonl").string(), "--record",
                 video.string()}),
            0);

  double fps = 0;
  EXPECT_EQ(read_video(video, &fps).size(), 3u);
  EXPECT_DOUBLE_EQ(fps, 12);
}

TEST(PipelineCliRecord, AnImageSequenceRecordsAtThirtyFps) {
  auto video = temp_path("run.avi");
  ASSERT_EQ(cli({frames(2), "--model", MODEL, "--timing", temp_path("t.jsonl").string(), "--record",
                 video.string()}),
            0);
  double fps = 0;
  read_video(video, &fps);
  EXPECT_DOUBLE_EQ(fps, 30);
}

TEST(PipelineCliRecord, AVideoThatCantBeWrittenIsAnError) {
  auto dir = temp_path("x").parent_path();  // a folder, not a file
  std::string err;
  EXPECT_EQ(cli({frames(1), "--model", MODEL, "--timing", temp_path("t.jsonl").string(), "--record", dir.string()},
                &err),
            1);
  EXPECT_NE(err.find("can't record"), std::string::npos) << err;
}

TEST(PipelineCliRecord, ShowAndRecordDontGoWithDump) {
  std::string err;
  EXPECT_EQ(cli({frames(1), "--model", MODEL, "--dump", temp_path("d.jsonl").string(), "--record",
                 temp_path("r.avi").string()},
                &err),
            2);
  EXPECT_NE(err.find("--dump"), std::string::npos) << err;
  EXPECT_EQ(cli({frames(1), "--model", MODEL, "--dump", temp_path("d.jsonl").string(), "--show"}), 2);
}
