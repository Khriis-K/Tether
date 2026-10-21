// The HUD view: the pipeline's draw step when there's something to draw on.
// Draws the HUD on each frame, records it, and shows it in a window, where the
// presenter can pause (Space) and quit (q or Esc).
#pragma once

#include <functional>
#include <opencv2/core.hpp>
#include <vector>

#include "tether/formats.hpp"

namespace tether {

// Where frames are shown and keys come from: GuiScreen (gui_screen.hpp), or a
// scripted stand-in in the tests.
class Screen {
 public:
  virtual ~Screen() = default;
  virtual void show(const cv::Mat& frame) = 0;
  // Waits up to `ms` for a key; -1 when none was pressed.
  virtual int wait_key(int ms) = 0;
  // True once the viewer has closed the window.
  virtual bool closed() const = 0;
};

// Takes each drawn frame, once, in order.
using Recorder = std::function<void(const cv::Mat&)>;

class HudView {
 public:
  // Either may be empty: no screen draws and records only, no recorder only shows.
  HudView(Screen* screen, Recorder recorder);

  // Draws `tracks` on a copy of `frame`, records it and shows it. Space holds
  // this frame on screen until Space again; while paused the pipeline waits.
  // False once the viewer quits or closes the window.
  bool show(const cv::Mat& frame, const std::vector<Track>& tracks);

 private:
  Screen* screen_;
  Recorder recorder_;
};

}  // namespace tether
