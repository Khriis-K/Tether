// The stats panel next to the video: what it lists, and that it keeps to the
// visual design's colours. Drawn with Dear ImGui into a context with no window,
// and checked through the vertices it would send to the GPU.

#include <gtest/gtest.h>

#include <imgui.h>

#include <set>
#include <vector>

#include "tether/hud.hpp"
#include "tether/panel.hpp"

using namespace tether;

namespace {

Track track(int id, int cls, const char* status, int missing = 0) {
  return {id, {0, 0, 10, 10}, cls, status, {0, 0, 10, 10}, missing, {}};
}

LiveSummary summary() {
  LiveSummary s{};
  s.frames = 2;
  s.frame_time = Stat{25, 30};
  s.fps = 40;
  s.end_to_end = {20, 22};
  s.steps = {Stat{2, 3}, Stat{1, 1}, Stat{15, 18}, Stat{0.5, 1}, Stat{1.5, 2}};
  return s;
}

ImU32 rgba(const cv::Scalar& bgr) {
  return IM_COL32(static_cast<int>(bgr[2]), static_cast<int>(bgr[1]), static_cast<int>(bgr[0]), 255);
}

// Draws one panel in a window-less ImGui context; returns every vertex colour
// with its alpha set to opaque (text edges and fills fade, the hue doesn't).
class HeadlessPanel {
 public:
  HeadlessPanel() {
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini left behind
    fonts_ = load_panel_fonts(*ImGui::GetIO().Fonts, 1.0f);
    unsigned char* pixels;
    int w, h;
    ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
    ImGui::GetIO().DisplaySize = {800, 600};
  }
  ~HeadlessPanel() { ImGui::DestroyContext(); }

  std::set<ImU32> colours(const PanelData& data) {
    ImGui::NewFrame();
    draw_panel(data, fonts_, {400, 0}, {400, 600});
    ImGui::Render();
    std::set<ImU32> seen;
    const ImDrawData* draw = ImGui::GetDrawData();
    for (const ImDrawList* list : draw->CmdLists)
      for (const ImDrawVert& v : list->VtxBuffer) seen.insert(v.col | IM_COL32_A_MASK);
    return seen;
  }

 private:
  PanelFonts fonts_;
};

}  // namespace

TEST(ClassName, NamesTheCocoClasses) {
  EXPECT_EQ(class_name(0), "person");
  EXPECT_EQ(class_name(2), "car");
  EXPECT_EQ(class_name(79), "toothbrush");
  EXPECT_EQ(class_name(80), "class 80");
  EXPECT_EQ(class_name(-1), "class -1");
}

TEST(TrackRows, ListTrackedAndSearchingTracksByIdButNotTentativeOnes) {
  auto rows = track_rows({track(7, 2, "searching", 4), track(3, 0, "tracked"), track(9, 0, "tentative")});

  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0].id, 3);
  EXPECT_EQ(rows[0].class_name, "person");
  EXPECT_EQ(rows[0].status, "tracked");
  EXPECT_FALSE(rows[0].searching);
  EXPECT_EQ(rows[1].id, 7);
  EXPECT_EQ(rows[1].class_name, "car");
  EXPECT_EQ(rows[1].status, "missing 4");
  EXPECT_TRUE(rows[1].searching);
}

TEST(StepBars, GiveEachStepItsMeanAndShareOfTheTotal) {
  auto bars = step_bars(summary());

  ASSERT_EQ(bars.size(), 5u);
  const char* names[] = {"capture", "prep", "detect", "track", "draw"};
  for (int i = 0; i < 5; ++i) EXPECT_EQ(bars[i].name, names[i]);
  EXPECT_DOUBLE_EQ(bars[2].ms, 15);
  EXPECT_DOUBLE_EQ(bars[2].share, 0.75);  // of 20 ms in all
  EXPECT_DOUBLE_EQ(bars[3].share, 0.025);
}

TEST(StepBars, AllZeroStepsGiveEmptyBars) {
  LiveSummary s{};
  for (const auto& bar : step_bars(s)) EXPECT_DOUBLE_EQ(bar.share, 0);
}

TEST(FormatMs, ShowsOneDecimal) {
  EXPECT_EQ(format_ms(12.345), "12.3 ms");
  EXPECT_EQ(format_ms(0.04), "0.0 ms");
}

TEST(Panel, DrawsBeforeAnyTimingHasArrived) {
  HeadlessPanel panel;
  EXPECT_FALSE(panel.colours({std::nullopt, {}, {{"runtime", "ONNX Runtime"}}}).empty());
}

TEST(Panel, UsesTheAccentOnlyForSearchingTracks) {
  HeadlessPanel panel;
  Settings settings{{"runtime", "ONNX Runtime (CPU)"}, {"pipeline", "sequential"}};

  auto calm = panel.colours({summary(), {track(1, 0, "tracked")}, settings});
  auto searching = panel.colours({summary(), {track(1, 0, "tracked"), track(2, 0, "searching", 3)}, settings});

  EXPECT_FALSE(calm.count(rgba(ACCENT)));
  EXPECT_TRUE(searching.count(rgba(ACCENT)));
}

TEST(Panel, KeepsToThePaletteAndEachTracksOwnColour) {
  HeadlessPanel panel;
  auto seen = panel.colours({summary(), {track(1, 0, "tracked"), track(2, 0, "searching", 3)}, {{"a", "b"}}});

  std::set<ImU32> allowed = {rgba(NEAR_BLACK), rgba(OFF_WHITE), rgba(MUTED), rgba(RULE), rgba(ACCENT),
                             rgba(track_color(1)), rgba(track_color(2))};
  for (ImU32 c : seen) EXPECT_TRUE(allowed.count(c)) << std::hex << "unexpected colour 0x" << c;
  EXPECT_TRUE(seen.count(rgba(NEAR_BLACK)));  // the background
  EXPECT_TRUE(seen.count(rgba(OFF_WHITE)));   // the text
}
