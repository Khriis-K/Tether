#include "tether/panel.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

#include "tether/hud.hpp"
#include "tether/panel_fonts.hpp"

namespace tether {

namespace {

constexpr std::array<const char*, 80> COCO = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",          "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",          "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",      "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",       "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",         "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",      "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",         "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",     "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

constexpr const char* STEP_NAMES[] = {"capture", "prep", "detect", "track", "draw"};

ImU32 rgba(const cv::Scalar& bgr) {
  return IM_COL32(static_cast<int>(bgr[2]), static_cast<int>(bgr[1]), static_cast<int>(bgr[0]), 255);
}

ImFont* add_font(ImFontAtlas& atlas, const FontFile& file, float size) {
  ImFontConfig config;
  config.FontDataOwnedByAtlas = false;  // the bytes are the program's, not the atlas's to free
  // ImGui only reads the bytes, but its signature takes them as mutable.
  return atlas.AddFontFromMemoryTTF(const_cast<unsigned char*>(file.data), static_cast<int>(file.size), size,
                                    &config);
}

// Muted text in Plex Mono: labels and headings.
void label(const PanelFonts& fonts, const char* text) {
  ImGui::PushFont(fonts.mono);
  ImGui::PushStyleColor(ImGuiCol_Text, rgba(MUTED));
  ImGui::TextUnformatted(text);
  ImGui::PopStyleColor();
  ImGui::PopFont();
}

// `text` on the current line, flush with the right edge, in the current font and colour.
void right_aligned(const std::string& text) {
  ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize(text.c_str()).x);
  ImGui::TextUnformatted(text.c_str());
}

// A small caps-style heading over each section.
void heading(const PanelFonts& fonts, const char* text) {
  ImGui::Dummy({0, ImGui::GetStyle().ItemSpacing.y * 0.5f});
  label(fonts, text);
  ImVec2 at = ImGui::GetCursorScreenPos();
  ImGui::GetWindowDrawList()->AddLine(at, {at.x + ImGui::GetContentRegionAvail().x, at.y}, rgba(RULE));
  ImGui::Dummy({0, 2});
}

// A label on the left and a value on the right, in `value_font`.
void labelled_value(const PanelFonts& fonts, const char* name, const std::string& value, ImFont* value_font) {
  label(fonts, name);
  ImGui::PushFont(value_font);
  right_aligned(value);
  ImGui::PopFont();
}

void draw_timing(const PanelFonts& fonts, const std::optional<LiveSummary>& stats) {
  const char* none = "-";
  ImGui::PushFont(fonts.big);
  char fps[16];
  if (stats && stats->fps) std::snprintf(fps, sizeof fps, "%.1f", *stats->fps);
  ImGui::TextUnformatted(stats && stats->fps ? fps : none);
  ImGui::PopFont();
  label(fonts, ("FPS, last " + std::to_string(stats ? stats->frames : 0) + " frames").c_str());

  heading(fonts, "FRAME TIME");
  bool timed = stats && stats->frame_time;
  labelled_value(fonts, "average", timed ? format_ms(stats->frame_time->mean) : none, fonts.mono);
  labelled_value(fonts, "worst 1% (P99)", timed ? format_ms(stats->frame_time->p99) : none, fonts.mono);
  labelled_value(fonts, "end-to-end delay", stats ? format_ms(stats->end_to_end.mean) : none, fonts.mono);

  heading(fonts, "STEPS (average)");
  if (!stats) return;
  // One line a step: name, a bar for its share of the time, then the time.
  ImGui::PushFont(fonts.mono);
  float left = ImGui::GetCursorPosX(), label_w = ImGui::CalcTextSize("capture ").x;
  float value_w = ImGui::CalcTextSize(" 000.0 ms").x;
  float bar_w = std::max(ImGui::GetContentRegionAvail().x - label_w - value_w, 1.0f);
  float line = ImGui::GetTextLineHeight(), bar_h = line * 0.45f;
  for (const auto& bar : step_bars(*stats)) {
    label(fonts, bar.name.c_str());
    ImGui::SameLine(left + label_w);
    ImVec2 at = ImGui::GetCursorScreenPos();
    at.y += (line - bar_h) / 2;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(at, {at.x + bar_w, at.y + bar_h}, rgba(RULE));
    draw->AddRectFilled(at, {at.x + bar_w * static_cast<float>(bar.share), at.y + bar_h}, rgba(MUTED));
    right_aligned(format_ms(bar.ms));
  }
  ImGui::PopFont();
}

// Fills the height left, less `keep` for what comes after; scrolls when there are more.
void draw_tracks(const PanelFonts& fonts, const std::vector<Track>& tracks, float keep) {
  auto rows = track_rows(tracks);
  char title[32];
  std::snprintf(title, sizeof title, "TRACKS (%zu)", rows.size());
  heading(fonts, title);
  // At least a few rows: on a short window the panel scrolls instead (mouse wheel).
  float least = 3 * ImGui::GetTextLineHeightWithSpacing();
  ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, ImGui::GetFontSize() * 0.4f);
  ImGui::BeginChild("tracks", {0, std::max(ImGui::GetContentRegionAvail().y - keep, least)});
  ImGui::PushFont(fonts.mono);
  float swatch = ImGui::GetTextLineHeight() * 0.6f;
  for (const auto& row : rows) {
    ImVec2 at = ImGui::GetCursorScreenPos();
    float inset = (ImGui::GetTextLineHeight() - swatch) / 2;
    ImGui::GetWindowDrawList()->AddRectFilled({at.x, at.y + inset}, {at.x + swatch, at.y + inset + swatch},
                                              rgba(track_color(row.id)));
    ImGui::Dummy({swatch, ImGui::GetTextLineHeight()});
    ImGui::SameLine();
    ImGui::Text("%4d", row.id);
    ImGui::SameLine();
    ImGui::PushFont(fonts.text);
    ImGui::TextUnformatted(row.class_name.c_str());
    ImGui::PopFont();
    ImGui::PushStyleColor(ImGuiCol_Text, row.searching ? rgba(ACCENT) : rgba(MUTED));
    right_aligned(row.status);
    ImGui::PopStyleColor();
  }
  ImGui::PopFont();
  ImGui::EndChild();
  ImGui::PopStyleVar();
}

void draw_settings(const PanelFonts& fonts, const Settings& settings) {
  heading(fonts, "SETTINGS");
  for (const auto& [name, value] : settings) labelled_value(fonts, name.c_str(), value, fonts.text);
}

}  // namespace

std::string class_name(int cls) {
  if (cls >= 0 && cls < static_cast<int>(COCO.size())) return COCO[cls];
  return "class " + std::to_string(cls);
}

std::vector<TrackRow> track_rows(const std::vector<Track>& tracks) {
  std::vector<TrackRow> rows;
  for (const auto& t : tracks) {
    if (t.status == "tracked") rows.push_back({t.id, class_name(t.cls), "tracked", false});
    if (t.status == "searching")
      rows.push_back({t.id, class_name(t.cls), "missing " + std::to_string(t.frames_since_seen), true});
  }
  std::sort(rows.begin(), rows.end(), [](const TrackRow& a, const TrackRow& b) { return a.id < b.id; });
  return rows;
}

std::vector<StepBar> step_bars(const LiveSummary& stats) {
  double total = 0;
  for (const auto& step : stats.steps) total += step.mean;
  std::vector<StepBar> bars;
  for (size_t i = 0; i < stats.steps.size(); ++i)
    bars.push_back({STEP_NAMES[i], stats.steps[i].mean, total > 0 ? stats.steps[i].mean / total : 0});
  return bars;
}

std::string format_ms(double ms) {
  char text[32];
  std::snprintf(text, sizeof text, "%.1f ms", ms);
  return text;
}

PanelFonts load_panel_fonts(ImFontAtlas& atlas, float scale) {
  PanelFonts fonts;
  fonts.text = add_font(atlas, PLEX_SANS_REGULAR, 16 * scale);  // the first font is ImGui's default
  fonts.mono = add_font(atlas, PLEX_MONO_REGULAR, 15 * scale);
  fonts.big = add_font(atlas, PLEX_MONO_SEMIBOLD, 44 * scale);
  return fonts;
}

void draw_panel(const PanelData& data, const PanelFonts& fonts, ImVec2 pos, ImVec2 size) {
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, rgba(NEAR_BLACK));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, rgba(NEAR_BLACK));
  ImGui::PushStyleColor(ImGuiCol_Border, rgba(RULE));
  ImGui::PushStyleColor(ImGuiCol_Text, rgba(OFF_WHITE));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, rgba(NEAR_BLACK));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, rgba(RULE));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, rgba(MUTED));
  ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, rgba(MUTED));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {fonts.text->FontSize, fonts.text->FontSize * 0.75f});
  ImGui::Begin("stats", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImGui::PushFont(fonts.text);
  draw_timing(fonts, data.stats);
  // The settings' heading and rows, so the track list leaves room for them.
  float settings_h = (static_cast<float>(data.settings.size()) + 2) * ImGui::GetTextLineHeightWithSpacing() +
                     2 * ImGui::GetStyle().ItemSpacing.y;
  draw_tracks(fonts, data.tracks, settings_h);
  draw_settings(fonts, data.settings);
  ImGui::PopFont();
  ImGui::End();
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor(8);
}

}  // namespace tether
