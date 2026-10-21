#include "tether/gui_screen.hpp"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>

#include "tether/hud.hpp"

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F  // OpenGL 1.2; Windows' gl.h stops at 1.1
#endif

namespace tether {

namespace {

constexpr float PANEL_WIDTH = 360;  // at 96 DPI
constexpr int ESC = 27;

// The keys HudView acts on, as the characters it expects.
void on_key(GLFWwindow* window, int key, int, int action, int) {
  if (action != GLFW_PRESS) return;
  auto* keys = static_cast<std::deque<int>*>(glfwGetWindowUserPointer(window));
  if (key == GLFW_KEY_SPACE) keys->push_back(' ');
  if (key == GLFW_KEY_Q) keys->push_back('q');
  if (key == GLFW_KEY_ESCAPE) keys->push_back(ESC);
}

}  // namespace

GuiScreen::GuiScreen(const std::string& title, PanelSource panel) : panel_(std::move(panel)) {
  if (!glfwInit()) throw std::runtime_error("can't open a window (GLFW didn't start)");
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);  // shown once it's sized to the video
  window_ = glfwCreateWindow(1280, 720, title.c_str(), nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("can't open a window (no OpenGL 3)");
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(0);  // no waiting for the monitor's refresh: that would cap FPS and pad the draw time
  glfwSetWindowUserPointer(window_, &keys_);
  glfwSetKeyCallback(window_, on_key);  // before ImGui's, which then passes keys on to it
  float yscale;
  glfwGetWindowContentScale(window_, &scale_, &yscale);

  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini left behind
  ImGui::GetStyle().ScaleAllSizes(scale_);
  fonts_ = load_panel_fonts(*ImGui::GetIO().Fonts, scale_);
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  if (!ImGui_ImplOpenGL3_Init("#version 130")) {
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
    throw std::runtime_error("can't open a window (ImGui couldn't set up OpenGL)");
  }

  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

GuiScreen::~GuiScreen() {
  glDeleteTextures(1, &texture_);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GuiScreen::fit_to(cv::Size frame) {
  int width = frame.width + static_cast<int>(PANEL_WIDTH * scale_), height = frame.height;
  int x, y, area_w, area_h;
  glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), &x, &y, &area_w, &area_h);
  // Shrinks to fit the screen, leaving room for the title bar.
  double fit = std::min({1.0, 0.95 * area_w / width, 0.9 * area_h / height});
  glfwSetWindowSize(window_, static_cast<int>(width * fit), static_cast<int>(height * fit));
  glfwShowWindow(window_);
  fitted_ = true;
}

void GuiScreen::show(const cv::Mat& frame) {
  if (!fitted_) fit_to(frame.size());
  cv::cvtColor(frame, rgb_, cv::COLOR_BGR2RGB);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_.cols, rgb_.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_.data);
  render();
  glfwSwapBuffers(window_);
}

void GuiScreen::render() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImVec2 window = ImGui::GetIO().DisplaySize;
  float panel_width = std::min(PANEL_WIDTH * scale_, window.x);
  float video_w = window.x - panel_width, video_h = window.y;

  // The video, as large as fits its side while keeping its shape, centred.
  float fit = std::min(video_w / rgb_.cols, video_h / rgb_.rows);
  ImVec2 size{rgb_.cols * fit, rgb_.rows * fit};
  ImVec2 at{(video_w - size.x) / 2, (video_h - size.y) / 2};
  ImGui::GetBackgroundDrawList()->AddImage(static_cast<ImTextureID>(texture_), at, {at.x + size.x, at.y + size.y});
  draw_panel(panel_(), fonts_, {video_w, 0}, {panel_width, window.y});

  ImGui::Render();
  int fb_w, fb_h;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  glViewport(0, 0, fb_w, fb_h);
  glClearColor(static_cast<float>(NEAR_BLACK[2] / 255), static_cast<float>(NEAR_BLACK[1] / 255),
               static_cast<float>(NEAR_BLACK[0] / 255), 1);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GuiScreen::save_screenshot(const std::filesystem::path& path) {
  if (rgb_.empty()) throw std::runtime_error("no frame shown to save");
  render();  // into the back buffer, which isn't covered by other windows or a locked screen
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  cv::Mat pixels(height, width, CV_8UC3);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data);
  cv::flip(pixels, pixels, 0);  // OpenGL's rows run bottom-up
  cv::cvtColor(pixels, pixels, cv::COLOR_RGB2BGR);
  if (!cv::imwrite(path.string(), pixels)) throw std::runtime_error("can't write '" + path.string() + "'");
}

int GuiScreen::wait_key(int ms) {
  if (keys_.empty()) {
    if (ms <= 1) glfwPollEvents();
    else glfwWaitEventsTimeout(ms / 1000.0);
  }
  if (keys_.empty()) return -1;
  int key = keys_.front();
  keys_.pop_front();
  return key;
}

bool GuiScreen::closed() const { return glfwWindowShouldClose(window_) != 0; }

}  // namespace tether
