#include "tether/hud.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <string>

namespace tether {

namespace {

constexpr double GOLDEN_RATIO = 0.618033988749895;
// Track hues skip [ACCENT_BAND_END - ACCENT_BAND, ACCENT_BAND_END), around the accent's hue (0.07).
constexpr double ACCENT_BAND_END = 0.12;
constexpr double ACCENT_BAND = 0.1;
constexpr double SATURATION = 0.85;

constexpr int FONT = cv::FONT_HERSHEY_SIMPLEX;

// Sizes are in pixels for frames up to 800 wide, and grow with bigger frames
// so the HUD stays readable on 1080p video.
struct Style {
  int unit;
  int bracket() const { return 2 * unit; }
  int predicted() const { return unit; }
  int searching() const { return 2 * unit; }
  int trail() const { return 2 * unit; }
  int pad() const { return 3 * unit; }
  double label_scale() const { return 0.5 * unit; }
  double status_scale() const { return 0.6 * unit; }
};

Style style_for(const cv::Mat& frame) { return {std::max(1, frame.cols / 800)}; }

cv::Point to_point(double x, double y) { return {static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y))}; }

cv::Rect to_rect(const Box& b) {
  auto tl = to_point(b[0], b[1]), br = to_point(b[2], b[3]);
  return {tl, br};
}

void dashed_line(cv::Mat& img, cv::Point from, cv::Point to, const cv::Scalar& color, int thickness, int dash, int gap) {
  cv::Point2d d = to - from;
  double length = std::hypot(d.x, d.y);
  if (length == 0) return;
  cv::Point2d step = d / length;
  for (double start = 0; start < length; start += dash + gap) {
    double end = std::min(start + dash - 1, length);
    cv::Point2d a = cv::Point2d(from) + step * start, b = cv::Point2d(from) + step * end;
    cv::line(img, to_point(a.x, a.y), to_point(b.x, b.y), color, thickness, cv::LINE_8);
  }
}

// Every edge starts its dashes at a corner, so the corners are always marked.
void dashed_rect(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int thickness, int dash, int gap) {
  cv::Point tl = r.tl(), br = r.br(), tr(br.x, tl.y), bl(tl.x, br.y);
  dashed_line(img, tl, tr, color, thickness, dash, gap);
  dashed_line(img, tl, bl, color, thickness, dash, gap);
  dashed_line(img, br, tr, color, thickness, dash, gap);
  dashed_line(img, br, bl, color, thickness, dash, gap);
}

void corner_brackets(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int thickness) {
  int arm = std::max(4, std::min(r.width, r.height) / 4);
  cv::Point tl = r.tl(), br = r.br();
  for (cv::Point corner : {tl, cv::Point(br.x, tl.y), cv::Point(tl.x, br.y), br}) {
    int sx = corner.x == tl.x ? 1 : -1, sy = corner.y == tl.y ? 1 : -1;
    cv::line(img, corner, corner + cv::Point(sx * arm, 0), color, thickness, cv::LINE_8);
    cv::line(img, corner, corner + cv::Point(0, sy * arm), color, thickness, cv::LINE_8);
  }
}

// A filled box with text in it, for labels and badges. `dot` leaves room before
// the text for a round marker.
struct Chip {
  std::string text;
  double scale;
  cv::Size text_size, size;
  int lead;  // space before the text
};

Chip make_chip(const std::string& text, double scale, bool dot, const Style& s) {
  int baseline = 0;
  cv::Size text_size = cv::getTextSize(text, FONT, scale, s.unit, &baseline);
  int lead = dot ? text_size.height + s.pad() : 0;
  return {text, scale, text_size,
          {text_size.width + lead + 2 * s.pad(), text_size.height + baseline + 2 * s.pad()}, lead};
}

void draw_chip(cv::Mat& img, cv::Point origin, const Chip& c, const cv::Scalar& fill, const cv::Scalar& ink,
               const Style& s) {
  cv::rectangle(img, cv::Rect(origin, c.size), fill, cv::FILLED);
  cv::putText(img, c.text, origin + cv::Point(s.pad() + c.lead, s.pad() + c.text_size.height), FONT, c.scale, ink,
              s.unit, cv::LINE_AA);
}

// A label chip sitting just above `box`, or just inside its top when there's
// no room above. Kept inside the frame.
void label(cv::Mat& img, const cv::Rect& box, int line, const std::string& text, const cv::Scalar& fill,
           const Style& s) {
  Chip chip = make_chip(text, s.label_scale(), false, s);
  int top = box.y - line - 1 - chip.size.height;
  if (top < 0) top = box.y + line + 1;
  top = std::clamp(top, 0, std::max(0, img.rows - chip.size.height));
  int left = std::clamp(box.x, 0, std::max(0, img.cols - chip.size.width));
  draw_chip(img, {left, top}, chip, fill, NEAR_BLACK, s);
}

// The path as segments, each blended over the frame more strongly the newer it is.
void trail(cv::Mat& img, const std::vector<Point>& path, const cv::Scalar& color, int thickness) {
  cv::Rect frame(0, 0, img.cols, img.rows);
  for (size_t i = 1; i < path.size(); ++i) {
    cv::Point a = to_point(path[i - 1][0], path[i - 1][1]), b = to_point(path[i][0], path[i][1]);
    cv::Point margin(thickness + 1, thickness + 1);
    cv::Rect area = cv::Rect(cv::Point(std::min(a.x, b.x), std::min(a.y, b.y)) - margin,
                             cv::Point(std::max(a.x, b.x), std::max(a.y, b.y)) + margin) &
                    frame;
    if (area.empty()) continue;
    cv::Mat under = img(area), over = under.clone();
    cv::line(over, a - area.tl(), b - area.tl(), color, thickness, cv::LINE_AA);
    double alpha = static_cast<double>(i) / (path.size() - 1);
    cv::addWeighted(over, alpha, under, 1 - alpha, 0, under);
  }
}

// A status badge at `origin` (top left); returns where the next one goes.
cv::Point badge(cv::Mat& img, cv::Point origin, const std::string& text, bool dot, const Style& s) {
  Chip chip = make_chip(text, s.status_scale(), dot, s);
  draw_chip(img, origin, chip, NEAR_BLACK, OFF_WHITE, s);
  int r = chip.text_size.height / 2;
  if (dot) cv::circle(img, origin + cv::Point(s.pad() + r, chip.size.height / 2), r, ACCENT, cv::FILLED, cv::LINE_AA);
  return origin + cv::Point(chip.size.width + s.pad(), 0);
}

}  // namespace

double track_hue(int id) {
  double spread = std::fmod(id * GOLDEN_RATIO, 1.0);
  return std::fmod(ACCENT_BAND_END + (1 - ACCENT_BAND) * spread, 1.0);
}

cv::Scalar track_color(int id) {
  // HSV to RGB at full value, as Python's colorsys does.
  double h = track_hue(id) * 6;
  int sector = static_cast<int>(h) % 6;
  double f = h - std::floor(h);
  double p = 1 - SATURATION, q = 1 - SATURATION * f, t = 1 - SATURATION * (1 - f);
  double r, g, b;
  switch (sector) {
    case 0: r = 1, g = t, b = p; break;
    case 1: r = q, g = 1, b = p; break;
    case 2: r = p, g = 1, b = t; break;
    case 3: r = p, g = q, b = 1; break;
    case 4: r = t, g = p, b = 1; break;
    default: r = 1, g = p, b = q; break;
  }
  return {std::round(b * 255), std::round(g * 255), std::round(r * 255)};
}

void draw_hud(cv::Mat& frame, const std::vector<Track>& tracks) {
  Style s = style_for(frame);
  // Trails first, so boxes and labels draw over them.
  for (const auto& t : tracks)
    if (t.status == "tracked") trail(frame, t.path, track_color(t.id), s.trail());
  for (const auto& t : tracks) {
    if (t.status == "tracked") {
      auto color = track_color(t.id);
      dashed_rect(frame, to_rect(t.predicted_box), color, s.predicted(), 4 * s.unit, 4 * s.unit);
      auto box = to_rect(t.box);
      corner_brackets(frame, box, color, s.bracket());
      label(frame, box, s.bracket(), std::to_string(t.id), color, s);
    } else if (t.status == "searching") {
      // Drawn at the prediction, the box the tracker searches around next frame:
      // the last seen box would stay put while the object moves on.
      auto box = to_rect(t.predicted_box);
      dashed_rect(frame, box, ACCENT, s.searching(), 10 * s.unit, 6 * s.unit);
      label(frame, box, s.searching(), std::to_string(t.id) + "  missing " + std::to_string(t.frames_since_seen),
            ACCENT, s);
    }
  }
}

void draw_status(cv::Mat& frame, bool paused, bool recording) {
  Style s = style_for(frame);
  cv::Point at(s.pad() * 3, s.pad() * 3);
  if (recording) at = badge(frame, at, "REC", true, s);
  if (paused) badge(frame, at, "PAUSED", false, s);
}

}  // namespace tether
