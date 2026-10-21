// The stats panel next to the video, drawn with Dear ImGui: a big live FPS
// number, then frame time, worst 1% (P99) and end-to-end delay, a bar per
// step, the current tracks and the speed settings in use.
//
// Visual design (docs/SPEC.md): near-black background, off-white text, IBM
// Plex Mono for numbers and labels, IBM Plex Sans for text, and the accent
// only where a track is searching. Each listed track carries its HUD colour.
#pragma once

#include <imgui.h>

#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tether/formats.hpp"
#include "tether/live_stats.hpp"

namespace tether {

inline const cv::Scalar MUTED{150, 146, 142};  // labels and step bars (BGR)
inline const cv::Scalar RULE{52, 48, 44};      // bar backgrounds and dividers

// What produced the numbers, as label and value: model, runtime, frame prep, pipeline mode.
using Settings = std::vector<std::pair<std::string, std::string>>;

struct PanelData {
  std::optional<LiveSummary> stats;  // empty until the first frame has been timed
  std::vector<Track> tracks;
  Settings settings;
};

struct TrackRow {
  int id;
  std::string class_name;
  std::string status;  // "tracked", or "missing K" while searching
  bool searching;
};

struct StepBar {
  std::string name;
  double ms;     // mean
  double share;  // of all steps' means together, 0-1
};

// The COCO name of a YOLO26 class ID; "class N" for any other ID.
std::string class_name(int cls);
// Tracked and searching tracks, by ID. Tentative tracks are left out, as on the video.
std::vector<TrackRow> track_rows(const std::vector<Track>& tracks);
std::vector<StepBar> step_bars(const LiveSummary& stats);
std::string format_ms(double ms);  // "12.3 ms"

struct PanelFonts {
  ImFont* text;  // IBM Plex Sans
  ImFont* mono;  // IBM Plex Mono: numbers and labels
  ImFont* big;   // IBM Plex Mono, large: the FPS number
};

// Adds the panel's fonts to `atlas`, sized by `scale` (1 at 96 DPI). They're
// built into the program, so nothing has to be found on disk.
PanelFonts load_panel_fonts(ImFontAtlas& atlas, float scale);

// Draws the panel as a fixed ImGui window at `pos`, `size` pixels.
void draw_panel(const PanelData& data, const PanelFonts& fonts, ImVec2 pos, ImVec2 size);

}  // namespace tether
