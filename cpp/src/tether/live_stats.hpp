// The stats panel's numbers: tether-timing's summary (docs/FORMATS.md, "Timing
// log"), worked out live over the most recent frames instead of a whole log.
// It reads the same step times the pipeline writes to the timing log.
#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "tether/formats.hpp"

namespace tether {

struct Stat {
  double mean, p99;  // milliseconds
};

struct LiveSummary {
  size_t frames;                 // how many recent frames these cover
  std::optional<Stat> frame_time;  // draw end to the next draw end; needs 2 frames
  std::optional<double> fps;       // 1000 / mean frame time
  Stat end_to_end;               // capture end to draw end
  std::array<Stat, 5> steps;     // capture, prep, detect, track, draw
};

// Worst 1% by nearest rank, as tether-timing: the value at rank ceil(0.99 n),
// counting from 1. Always one of `values`; under 100 values, the largest.
// `values` must not be empty.
double p99(std::vector<double> values);

class LiveStats {
 public:
  // 300 frames: 10 s at 30 FPS. Long enough for a P99 that isn't just the worst
  // frame, short enough that warm-up drops out soon after the start.
  static constexpr size_t DEFAULT_WINDOW = 300;

  explicit LiveStats(size_t window = DEFAULT_WINDOW);
  void add(const StepTimes& times);
  // Over the last `window` frames added; empty before the first.
  std::optional<LiveSummary> summary() const;

 private:
  size_t window_;
  std::deque<StepTimes> frames_;
};

}  // namespace tether
