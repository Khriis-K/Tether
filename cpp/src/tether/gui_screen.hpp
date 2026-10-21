// The --show window: the video with its HUD on the left and the stats panel
// beside it, drawn with Dear ImGui through OpenGL in a GLFW window.
#pragma once

#include <deque>
#include <filesystem>
#include <functional>
#include <opencv2/core.hpp>
#include <string>

#include "tether/hud_view.hpp"
#include "tether/panel.hpp"

struct GLFWwindow;

namespace tether {

class GuiScreen : public Screen {
 public:
  // Asked for the panel's contents each time a frame is shown.
  using PanelSource = std::function<PanelData()>;

  // Throws std::runtime_error when no window can be opened (no display, no OpenGL 3).
  GuiScreen(const std::string& title, PanelSource panel);
  ~GuiScreen() override;
  GuiScreen(const GuiScreen&) = delete;
  GuiScreen& operator=(const GuiScreen&) = delete;

  void show(const cv::Mat& frame) override;
  int wait_key(int ms) override;
  bool closed() const override;

  // Saves the window as last shown, video and panel, as an image file. Throws
  // std::runtime_error before the first frame or when the file can't be written.
  void save_screenshot(const std::filesystem::path& path);

 private:
  void fit_to(cv::Size frame);  // sizes the window to the first frame plus the panel
  void render();                // draws the shown frame and the panel, ready to swap in

  GLFWwindow* window_ = nullptr;
  PanelSource panel_;
  PanelFonts fonts_{};
  float scale_ = 1;  // the monitor's content scale: 1 at 96 DPI
  unsigned texture_ = 0;
  cv::Mat rgb_;  // the frame as the texture takes it
  bool fitted_ = false;
  std::deque<int> keys_;  // pressed and not yet read
};

}  // namespace tether
