// tether-pipeline --screenshot: saves the --show window, so it needs one. The
// picture itself needs a real window, so it's checked by running the app.

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;
using tether::test::temp_path;

TEST(PipelineCliScreenshot, NeedsTheWindow) {
  std::ostringstream out, err;
  int code = run_pipeline_cli({"video.mp4", "--screenshot", temp_path("shot.png").string()}, out, err);

  EXPECT_EQ(code, 2);
  EXPECT_NE(err.str().find("--screenshot saves the --show window"), std::string::npos) << err.str();
}

TEST(PipelineCliScreenshot, IsListedInTheHelp) {
  std::ostringstream out, err;
  ASSERT_EQ(run_pipeline_cli({"--help"}, out, err), 0);
  EXPECT_NE(out.str().find("--screenshot PNG"), std::string::npos);
}
