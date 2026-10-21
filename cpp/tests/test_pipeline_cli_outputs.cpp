// tether-pipeline creates the folders its output files go in (code review, #11).

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"

using tether::test::temp_path;

TEST(PipelineCliOutputs, DumpCreatesTheFolderItWritesTo) {
  auto dir = temp_path("frames").parent_path();
  cv::imwrite((dir / "f0.png").string(), cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0)));
  auto dets = dir / "new" / "folder" / "dets.jsonl";
  std::ostringstream out, err;

  int code = tether::run_pipeline_cli({(dir / "f%d.png").string(), "--model",
                                       std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx", "--dump", dets.string()},
                                      out, err);

  EXPECT_EQ(code, 0) << err.str();
  EXPECT_EQ(tether::load_detections(dets).size(), 1u);
}
