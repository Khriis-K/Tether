// The HUD drawing: what each track status looks like on a black frame. Checks
// pixels, not drawing calls, so the drawing code can change freely.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "tether/hud.hpp"

using namespace tether;

namespace {

cv::Mat black() { return cv::Mat(400, 400, CV_8UC3, cv::Scalar(0, 0, 0)); }

Track track(int id, const std::string& status, const Box& box, const Box& predicted, int missing = 0,
            std::vector<Point> path = {}) {
  return {id, box, 0, status, predicted, missing, std::move(path)};
}

bool is_color(const cv::Mat& f, int x, int y, const cv::Scalar& c, int tol = 40) {
  auto p = f.at<cv::Vec3b>(y, x);
  for (int i = 0; i < 3; ++i)
    if (std::abs(p[i] - c[i]) > tol) return false;
  return true;
}

bool is_black(const cv::Mat& f, int x, int y) { return f.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0); }

// Pixels of colour `c` on the horizontal line y, x0..x1.
int count_on_row(const cv::Mat& f, int y, int x0, int x1, const cv::Scalar& c) {
  int n = 0;
  for (int x = x0; x <= x1; ++x) n += is_color(f, x, y, c);
  return n;
}

// Pixels of colour `c` in the band rows y0..y1, columns x0..x1.
int count_in(const cv::Mat& f, int x0, int y0, int x1, int y1, const cv::Scalar& c) {
  int n = 0;
  for (int y = y0; y <= y1; ++y) n += count_on_row(f, y, x0, x1, c);
  return n;
}

int brightness(const cv::Mat& f, int x, int y) {
  auto p = f.at<cv::Vec3b>(y, x);
  return std::max({p[0], p[1], p[2]});
}

double hue_of(const cv::Scalar& bgr) {
  cv::Mat px(1, 1, CV_8UC3, bgr), hsv;
  cv::cvtColor(px, hsv, cv::COLOR_BGR2HSV_FULL);
  return hsv.at<cv::Vec3b>(0, 0)[0] / 256.0;  // a fraction of the wheel
}

double hue_gap(double a, double b) {
  double d = std::abs(a - b);
  return std::min(d, 1 - d);
}

}  // namespace

TEST(TrackColor, NeighbouringIdsAreFarApartOnTheWheel) {
  for (int id = 1; id < 50; ++id) EXPECT_GT(hue_gap(track_hue(id), track_hue(id + 1)), 0.25) << id;
}

TEST(TrackColor, AnyNumberOfTracksIsSpreadEvenly) {
  // An even split, knowing N in advance, would put hues 1/N apart. IDs arrive
  // one at a time, so aim for never bunching closer than a third of that.
  for (int n = 2; n <= 60; ++n) {
    std::vector<double> hues;
    for (int id = 1; id <= n; ++id) hues.push_back(track_hue(id));
    std::sort(hues.begin(), hues.end());
    double closest = 1 - hues.back() + hues.front();
    for (size_t i = 1; i < hues.size(); ++i) closest = std::min(closest, hues[i] - hues[i - 1]);
    EXPECT_GE(closest, 0.3 / n) << n << " tracks";
  }
}

TEST(TrackColor, NoTrackLooksLikeTheSearchingAccent) {
  for (int id = 1; id <= 200; ++id) EXPECT_GT(hue_gap(hue_of(track_color(id)), hue_of(ACCENT)), 0.04) << id;
}

TEST(TrackColor, IsTheHueAsABrightBgrColour) {
  for (int id : {1, 2, 7}) {
    EXPECT_NEAR(hue_of(track_color(id)), track_hue(id), 0.01) << id;
    EXPECT_GE(std::max({track_color(id)[0], track_color(id)[1], track_color(id)[2]}), 230) << id;
  }
}

TEST(DrawHud, TentativeTracksAreNotDrawn) {
  auto f = black();
  draw_hud(f, {track(1, "tentative", {40, 40, 160, 160}, {50, 50, 170, 170}, 0, {{100, 100}})});
  EXPECT_EQ(cv::countNonZero(f.reshape(1)), 0);
}

TEST(DrawHud, ATrackedObjectGetsCornerBracketsNotAFullBox) {
  auto f = black();
  auto c = track_color(3);
  draw_hud(f, {track(3, "tracked", {40, 140, 160, 260}, {240, 240, 360, 360})});

  for (auto [x, y] : {std::pair{40, 140}, {160, 140}, {40, 260}, {160, 260}}) EXPECT_TRUE(is_color(f, x, y, c)) << x;
  EXPECT_TRUE(is_color(f, 50, 140, c));      // along the top from a corner
  EXPECT_TRUE(is_color(f, 40, 150, c));      // down the side from a corner
  EXPECT_TRUE(is_black(f, 100, 140));        // middle of the top edge: open
  EXPECT_TRUE(is_black(f, 40, 200));         // middle of the side: open
  EXPECT_TRUE(is_black(f, 100, 200));        // inside: the object stays visible
}

TEST(DrawHud, ATrackedObjectsBracketsAreSolid) {
  auto f = black();
  auto c = track_color(3);
  draw_hud(f, {track(3, "tracked", {40, 140, 160, 260}, {240, 240, 360, 360})});
  // The first stretch of the top edge from the corner has no gaps.
  EXPECT_EQ(count_on_row(f, 140, 40, 55, c), 16);
}

TEST(DrawHud, ATrackedObjectHasItsIdLabelAboveTheBox) {
  auto f = black();
  draw_hud(f, {track(3, "tracked", {40, 140, 160, 260}, {240, 240, 360, 360})});
  cv::Mat above = f(cv::Rect(40, 110, 80, 28));
  EXPECT_GT(cv::countNonZero(above.reshape(1)), 50);

  auto g = black();
  draw_hud(g, {track(48, "tracked", {40, 140, 160, 260}, {240, 240, 360, 360})});
  EXPECT_GT(cv::norm(f(cv::Rect(40, 110, 80, 28)), g(cv::Rect(40, 110, 80, 28))), 0);  // the ID is in it
}

TEST(DrawHud, TheLabelStaysOnScreenForABoxAtTheTop) {
  auto f = black();
  draw_hud(f, {track(3, "tracked", {40, 0, 160, 120}, {240, 240, 360, 360})});
  // Inside the box's top, clear of the brackets themselves.
  EXPECT_GT(cv::countNonZero(f(cv::Rect(45, 5, 75, 25)).reshape(1)), 50);
}

TEST(DrawHud, ATrackedObjectsPredictedBoxIsDashedInItsColour) {
  auto f = black();
  auto c = track_color(3);
  draw_hud(f, {track(3, "tracked", {40, 140, 160, 260}, {240, 240, 360, 360})});
  int on = count_on_row(f, 240, 250, 350, c);
  EXPECT_GT(on, 30);   // drawn along the whole edge...
  EXPECT_LT(on, 80);   // ...with gaps
  EXPECT_GT(count_on_row(f, 300, 240, 240, c) + count_on_row(f, 320, 240, 240, c) +
                count_on_row(f, 340, 240, 240, c) + count_on_row(f, 280, 240, 240, c),
            0);        // and down the sides
}

TEST(DrawHud, ASearchingObjectIsAnAccentDashedBoxAtItsPredictionWithTheFramesMissing) {
  auto f = black();
  draw_hud(f, {track(5, "searching", {40, 140, 160, 260}, {240, 240, 360, 360}, 7)});

  int on = count_on_row(f, 240, 250, 350, ACCENT) + count_on_row(f, 241, 250, 350, ACCENT);
  EXPECT_GT(on, 30);
  EXPECT_LT(on, 2 * 101);  // with gaps
  EXPECT_EQ(count_in(f, 0, 0, 399, 399, track_color(5)), 0);  // no track colour: no brackets, no trail
  EXPECT_TRUE(is_black(f, 40, 140));  // nothing at the last seen box

  auto g = black();
  draw_hud(g, {track(5, "searching", {40, 140, 160, 260}, {240, 240, 360, 360}, 12)});
  cv::Rect label(240, 205, 150, 33);
  EXPECT_GT(cv::countNonZero(f(label).reshape(1)), 50);
  EXPECT_GT(cv::norm(f(label), g(label)), 0);  // the missing count is in the label
}

TEST(DrawHud, SearchingIsHeavierThanAPrediction) {
  // Status reads from line style alone: a prediction is thin dashes, searching thick ones.
  auto pred = black(), search = black();
  draw_hud(pred, {track(5, "tracked", {10, 10, 20, 20}, {240, 240, 360, 360})});
  draw_hud(search, {track(5, "searching", {10, 10, 20, 20}, {240, 240, 360, 360}, 1)});
  auto band = [](const cv::Mat& f) {
    cv::Mat gray;
    cv::cvtColor(f(cv::Rect(250, 239, 101, 6)), gray, cv::COLOR_BGR2GRAY);  // the top edge, below any label
    return cv::countNonZero(gray);
  };
  EXPECT_GT(band(search), 1.5 * band(pred));
}

TEST(DrawHud, TheTrailFadesFromTheNewestPointBackToTheOldest) {
  auto f = black();
  std::vector<Point> path = {{20, 380}, {110, 380}, {200, 380}, {290, 380}, {380, 380}};  // oldest first
  draw_hud(f, {track(2, "tracked", {340, 20, 390, 70}, {340, 20, 390, 70}, 0, path)});

  int old_end = brightness(f, 40, 380), middle = brightness(f, 200, 380), new_end = brightness(f, 360, 380);
  EXPECT_GT(old_end, 0);
  EXPECT_LT(old_end, middle);
  EXPECT_LT(middle, new_end);
  EXPECT_TRUE(is_color(f, 360, 380, track_color(2), 60));
}

TEST(DrawHud, DrawsEveryTrack) {
  auto f = black();
  draw_hud(f, {track(1, "tracked", {20, 140, 100, 220}, {20, 300, 100, 380}),
               track(2, "tracked", {220, 140, 300, 220}, {220, 300, 300, 380})});
  EXPECT_TRUE(is_color(f, 20, 140, track_color(1)));
  EXPECT_TRUE(is_color(f, 220, 140, track_color(2)));
}

TEST(DrawStatus, NothingWhenRunningAndNotRecording) {
  auto f = black();
  draw_status(f, false, false);
  EXPECT_EQ(cv::countNonZero(f.reshape(1)), 0);
}

TEST(DrawStatus, PausedAndRecordingEachShow) {
  auto paused = black(), recording = black(), both = black();
  draw_status(paused, true, false);
  draw_status(recording, false, true);
  draw_status(both, true, true);
  EXPECT_GT(cv::countNonZero(paused.reshape(1)), 0);
  EXPECT_GT(count_in(recording, 0, 0, 399, 399, ACCENT), 0);  // the live indicator is the accent
  EXPECT_GT(cv::norm(paused, recording), 0);
  EXPECT_GT(cv::countNonZero(both.reshape(1)), cv::countNonZero(paused.reshape(1)));
}
