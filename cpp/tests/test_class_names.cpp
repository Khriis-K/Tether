// Class names for the stats panel: from --class-names, else the model's ONNX
// metadata, else COCO's. The stand-in models are in tests/onnx/make_models.py.

#include <gtest/gtest.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/class_names.hpp"
#include "tether/panel.hpp"
#include "tether/pipeline_cli.hpp"

using namespace tether;
using tether::test::temp_path;

namespace {

const std::string DRONE_MODEL = std::string(TETHER_TEST_DATA) + "/drone_736x1280.onnx";  // names {0: 'drone'}
const std::string NAMELESS_MODEL = std::string(TETHER_TEST_DATA) + "/fixed_736x1280.onnx";  // no metadata

int cli(std::vector<std::string> args, std::string* err_text = nullptr) {
  std::ostringstream out, err;
  int code = run_pipeline_cli(args, out, err);
  if (err_text) *err_text = err.str();
  return code;
}

std::string one_frame() {
  auto path = temp_path("f0.png");
  cv::imwrite(path.string(), cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0)));
  return (path.parent_path() / "f%d.png").string();
}

// The text one panel draws, from ImGui's log of every item it renders.
std::string panel_text(const PanelData& data) {
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  PanelFonts fonts = load_panel_fonts(*ImGui::GetIO().Fonts, 1.0f);
  unsigned char* pixels;
  int w, h;
  ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
  ImGui::GetIO().DisplaySize = {800, 600};
  ImGui::NewFrame();
  ImGui::LogToBuffer();
  draw_panel(data, fonts, {400, 0}, {400, 600});
  std::string text = GImGui->LogBuffer.c_str();
  ImGui::LogFinish();
  ImGui::Render();
  ImGui::DestroyContext();
  return text;
}

}  // namespace

TEST(CocoNames, AreTheEightyYolo26ClassesInOrder) {
  ASSERT_EQ(coco_names().size(), 80u);
  EXPECT_EQ(coco_names()[0], "person");
  EXPECT_EQ(coco_names()[4], "airplane");
  EXPECT_EQ(coco_names()[79], "toothbrush");
}

TEST(ParseNamesMetadata, ReadsUltralyticsDictOfIdsToNames) {
  EXPECT_EQ(parse_names_metadata("{0: 'drone'}"), ClassNames{"drone"});
  EXPECT_EQ(parse_names_metadata("{0: 'person', 1: 'traffic light', 2: \"it's\"}"),
            (ClassNames{"person", "traffic light", "it's"}));
}

TEST(ParseNamesMetadata, RefusesMalformedTextOrIdsThatArentZeroToN) {
  for (const char* bad : {"", "{}", "drone", "{0: drone}", "{0: 'drone'", "{1: 'drone'}", "{0: 'a', 2: 'b'}",
                          "{0: 'a', 0: 'b'}", "{0: 'a',, 1: 'b'}", "{0: 'a'} x"})
    EXPECT_THROW(parse_names_metadata(bad), std::invalid_argument) << bad;
}

TEST(ParseNamesFlag, SplitsCommaSeparatedNamesByClassId) {
  EXPECT_EQ(parse_names_flag("drone"), ClassNames{"drone"});
  EXPECT_EQ(parse_names_flag("drone,bird"), (ClassNames{"drone", "bird"}));
  for (const char* bad : {"", "drone,", ",drone", "a,,b"})
    EXPECT_THROW(parse_names_flag(bad), std::invalid_argument) << bad;
}

TEST(ModelClassNames, ComeFromTheOnnxMetadata) {
  EXPECT_EQ(model_class_names(DRONE_MODEL), ClassNames{"drone"});
}

TEST(ModelClassNames, AreAbsentForAModelWithoutThemOrAMissingFile) {
  EXPECT_EQ(model_class_names(NAMELESS_MODEL), std::nullopt);
  // A TensorRT run needs only the engine, so the .onnx may not be there.
  EXPECT_EQ(model_class_names(temp_path("gone.onnx")), std::nullopt);
}

TEST(ModelClassNames, AreAbsentForAnEngineGivenAsTheModel) {
  // --model X.engine runs on TensorRT; the engine has no metadata, and ONNX Runtime can't open it.
  auto engine = temp_path("m.engine");
  std::ofstream(engine) << "not an onnx file";
  EXPECT_EQ(model_class_names(engine), std::nullopt);
}

TEST(ClassNamesFor, TakeTheFlagThenTheModelThenCoco) {
  EXPECT_EQ(class_names_for(DRONE_MODEL, std::string("uav")), ClassNames{"uav"});
  EXPECT_EQ(class_names_for(DRONE_MODEL, std::nullopt), ClassNames{"drone"});
  EXPECT_EQ(class_names_for(NAMELESS_MODEL, std::nullopt), coco_names());
}

TEST(TrackRows, NameTracksWithTheGivenClassNames) {
  Track t{1, {0, 0, 10, 10}, 0, "tracked", {0, 0, 10, 10}, 0, {}};
  EXPECT_EQ(track_rows({t}, class_names_for(DRONE_MODEL, std::nullopt))[0].class_name, "drone");
  EXPECT_EQ(track_rows({t}, class_names_for(NAMELESS_MODEL, std::nullopt))[0].class_name, "person");
  EXPECT_EQ(class_name(1, ClassNames{"drone"}), "class 1");
}

TEST(DrawPanel, ListsEachTrackByItsClassName) {
  PanelData data;
  data.tracks = {{1, {0, 0, 10, 10}, 0, "tracked", {0, 0, 10, 10}, 0, {}}};
  EXPECT_NE(panel_text(data).find("person"), std::string::npos);

  data.class_names = class_names_for(DRONE_MODEL, std::nullopt);
  auto text = panel_text(data);
  EXPECT_NE(text.find("drone"), std::string::npos);
  EXPECT_EQ(text.find("person"), std::string::npos);
}

TEST(PanelData, ListsCocoNamesUnlessGivenOthers) { EXPECT_EQ(PanelData{}.class_names, coco_names()); }

TEST(PipelineCli, ClassNamesFlagMustNameEveryClass) {
  std::string err;
  EXPECT_EQ(cli({one_frame(), "--model", NAMELESS_MODEL, "--class-names", "drone,", "--max-frames", "1"}, &err), 2);
  EXPECT_NE(err.find("--class-names"), std::string::npos);
  EXPECT_EQ(cli({one_frame(), "--model", NAMELESS_MODEL, "--class-names", "drone", "--max-frames", "1",
                 "--timing", temp_path("t.jsonl").string()}),
            0);
}
