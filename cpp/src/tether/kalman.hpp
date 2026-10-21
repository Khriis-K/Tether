// Motion model: a constant-velocity Kalman filter over one track's box.
//
// State is the box center and size plus their velocities: (cx, cy, w, h, vcx,
// vcy, vw, vh). Noise scales with the box height, so a small far object and a
// large near one are judged by the same relative standards.
#pragma once

#include <array>

#include "tether/formats.hpp"

namespace tether {

inline constexpr double POSITION_NOISE = 1.0 / 20;  // std of measured and predicted position/size, per box height
inline constexpr double VELOCITY_NOISE = 1.0 / 10;  // std of per-frame change in center velocity, per box height
// Real boxes change size slowly, so size velocity gets far less noise than
// center velocity. Otherwise one smaller box (a partial detection) is taken as
// a fast shrink and the next prediction is an inside-out box.
inline constexpr double SIZE_VELOCITY_NOISE = 1.0 / 160;

class BoxFilter {
 public:
  explicit BoxFilter(const Box& box);

  Box box() const;  // the current estimate: after predict(), the prediction
  // Step one frame forward, first multiplying the velocities by `damping`. Uncertainty grows.
  void predict(double damping = 1.0);
  // Blend in a matched detection. Uncertainty shrinks.
  void update(const Box& box);

 private:
  std::array<double, 8> x_;
  std::array<std::array<double, 8>, 8> p_;
};

}  // namespace tether
