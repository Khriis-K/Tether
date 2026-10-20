// The HUD view, the pipeline's draw step: draws, records, shows, and handles
// the presenter's keys. A scripted screen stands in for the window.

#include <gtest/gtest.h>

#include <deque>
#include <opencv2/core.hpp>
#include <vector>

#include "tether/hud.hpp"
#include "tether/hud_view.hpp"

using namespace tether;

namespace {

// Hands out scripted keys (then no key), and can close after a number of key reads.
class FakeScreen : public Screen {
 public:
  explicit FakeScreen(std::deque<int> keys, int close_after_reads = -1)
      : keys_(std::move(keys)), close_after_(close_after_reads) {}
  void show(const cv::Mat& frame) override { shown.push_back(frame.clone()); }
  int wait_key(int) override {
    ++reads;
    if (keys_.empty()) return -1;
    int k = keys_.front();
    keys_.pop_front();
    return k;
  }
  bool closed() const override { return close_after_ >= 0 && reads >= close_after_; }

  std::vector<cv::Mat> shown;
  int reads = 0;
  size_t keys_left() const { return keys_.size(); }

 private:
  std::deque<int> keys_;
  int close_after_;
};

cv::Mat frame() { return cv::Mat(400, 400, CV_8UC3, cv::Scalar(30, 30, 30)); }

std::vector<Track> tracks() { return {{1, {40, 140, 160, 260}, 0, "tracked", {240, 240, 360, 360}, 0, {}}}; }

cv::Mat drawn() {
  auto f = frame();
  draw_hud(f, tracks());
  return f;
}

bool same(const cv::Mat& a, const cv::Mat& b) { return a.size() == b.size() && cv::norm(a, b) == 0; }

}  // namespace

TEST(HudView, ShowsTheFrameWithTheTracksDrawnAndLeavesTheInputAlone) {
  FakeScreen screen({});
  HudView view(&screen, nullptr);
  auto input = frame();

  EXPECT_TRUE(view.show(input, tracks()));

  ASSERT_EQ(screen.shown.size(), 1u);
  EXPECT_TRUE(same(screen.shown[0], drawn()));
  EXPECT_TRUE(same(input, frame()));
}

TEST(HudView, QOrEscStops) {
  for (int key : {int{'q'}, 27}) {
    FakeScreen screen({key});
    HudView view(&screen, nullptr);
    EXPECT_FALSE(view.show(frame(), tracks())) << key;
  }
}

TEST(HudView, OtherKeysCarryOn) {
  FakeScreen screen({'x'});
  HudView view(&screen, nullptr);
  EXPECT_TRUE(view.show(frame(), tracks()));
}

TEST(HudView, ClosingTheWindowStops) {
  FakeScreen screen({}, 1);
  HudView view(&screen, nullptr);
  EXPECT_FALSE(view.show(frame(), tracks()));
}

TEST(HudView, SpacePausesOnTheFrameUntilSpaceAgain) {
  FakeScreen screen({' ', -1, -1, ' '});
  HudView view(&screen, nullptr);

  EXPECT_TRUE(view.show(frame(), tracks()));

  EXPECT_EQ(screen.keys_left(), 0u);  // held the frame through every key
  ASSERT_GE(screen.shown.size(), 2u);
  EXPECT_FALSE(same(screen.shown[1], drawn()));  // marked as paused
  EXPECT_TRUE(same(screen.shown.back(), drawn()));  // and unmarked once it goes on

  view.show(frame(), tracks());  // resumed: the next frame doesn't wait
  EXPECT_EQ(screen.reads, 5);
}

TEST(HudView, QuittingWhilePausedStops) {
  FakeScreen screen({' ', -1, 'q', ' '});
  HudView view(&screen, nullptr);
  EXPECT_FALSE(view.show(frame(), tracks()));
  EXPECT_EQ(screen.keys_left(), 1u);
}

TEST(HudView, ClosingTheWindowWhilePausedStops) {
  FakeScreen screen({' '}, 3);  // then no keys at all: only the close can end the pause
  HudView view(&screen, nullptr);
  EXPECT_FALSE(view.show(frame(), tracks()));
}

TEST(HudView, RecordsEachDrawnFrameOnceWithoutTheStatusMarks) {
  FakeScreen screen({' ', -1, ' '});
  std::vector<cv::Mat> recorded;
  HudView view(&screen, [&](const cv::Mat& f) { recorded.push_back(f.clone()); });

  view.show(frame(), tracks());
  view.show(frame(), tracks());

  ASSERT_EQ(recorded.size(), 2u);  // a pause doesn't repeat frames in the recording
  EXPECT_TRUE(same(recorded[0], drawn()));
  EXPECT_FALSE(same(screen.shown.back(), drawn()));  // the screen shows it's recording
}

TEST(HudView, WithoutAScreenItOnlyRecordsAndNeverStops) {
  int recorded = 0;
  HudView view(nullptr, [&](const cv::Mat& f) {
    EXPECT_TRUE(same(f, drawn()));
    ++recorded;
  });
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(view.show(frame(), tracks()));
  EXPECT_EQ(recorded, 3);
}
