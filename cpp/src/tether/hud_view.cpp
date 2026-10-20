#include "tether/hud_view.hpp"

#include <opencv2/highgui.hpp>
#include <utility>

#include "tether/hud.hpp"

namespace tether {

namespace {

constexpr int PAUSE_KEY = ' ';
constexpr int ESC = 27;
constexpr int PAUSED_POLL_MS = 30;  // how often a paused view checks for keys

bool is_quit(int key) { return key == 'q' || key == ESC; }

}  // namespace

WindowScreen::WindowScreen(std::string title) : title_(std::move(title)) {
  cv::namedWindow(title_, cv::WINDOW_AUTOSIZE);
}

WindowScreen::~WindowScreen() { cv::destroyWindow(title_); }

void WindowScreen::show(const cv::Mat& frame) {
  cv::imshow(title_, frame);
  shown_ = true;
}

int WindowScreen::wait_key(int ms) {
  int key = cv::waitKey(ms);
  return key < 0 ? -1 : key & 0xFF;
}

bool WindowScreen::closed() const {
  return shown_ && cv::getWindowProperty(title_, cv::WND_PROP_VISIBLE) < 1;
}

HudView::HudView(Screen* screen, Recorder recorder) : screen_(screen), recorder_(std::move(recorder)) {}

bool HudView::show(const cv::Mat& frame, const std::vector<Track>& tracks) {
  cv::Mat drawn = frame.clone();
  draw_hud(drawn, tracks);
  if (recorder_) recorder_(drawn);  // before the status marks: they're for the presenter
  if (!screen_) return true;

  bool recording = static_cast<bool>(recorder_);
  auto show_marked = [&](bool paused) {
    if (!paused && !recording) {
      screen_->show(drawn);
      return;
    }
    cv::Mat marked = drawn.clone();
    draw_status(marked, paused, recording);
    screen_->show(marked);
  };

  show_marked(false);
  int key = screen_->wait_key(1);
  if (key == PAUSE_KEY) {
    for (;;) {
      if (screen_->closed()) return false;
      show_marked(true);
      key = screen_->wait_key(PAUSED_POLL_MS);
      if (key == PAUSE_KEY || is_quit(key)) break;
    }
    if (key == PAUSE_KEY) show_marked(false);
  }
  return !is_quit(key) && !screen_->closed();
}

}  // namespace tether
