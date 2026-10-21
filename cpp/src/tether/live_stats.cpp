#include "tether/live_stats.hpp"

#include <algorithm>
#include <numeric>

namespace tether {

namespace {

Stat stat(const std::vector<double>& values) {
  return {std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size()), p99(values)};
}

}  // namespace

double p99(std::vector<double> values) {
  size_t rank = (99 * values.size() + 99) / 100;  // ceil(0.99 n), in exact integer math
  std::nth_element(values.begin(), values.begin() + (rank - 1), values.end());
  return values[rank - 1];
}

LiveStats::LiveStats(size_t window) : window_(window) {}

void LiveStats::add(const StepTimes& times) {
  frames_.push_back(times);
  if (frames_.size() > window_) frames_.pop_front();
}

std::optional<LiveSummary> LiveStats::summary() const {
  if (frames_.empty()) return std::nullopt;
  LiveSummary s{};
  s.frames = frames_.size();

  std::vector<double> frame_times, delays;
  for (size_t i = 0; i < frames_.size(); ++i) {
    if (i > 0) frame_times.push_back(frames_[i].draw[1] - frames_[i - 1].draw[1]);
    delays.push_back(frames_[i].draw[1] - frames_[i].capture[1]);
  }
  if (!frame_times.empty()) {
    s.frame_time = stat(frame_times);
    s.fps = 1000 / s.frame_time->mean;
  }
  s.end_to_end = stat(delays);

  Span StepTimes::*steps[] = {&StepTimes::capture, &StepTimes::prep, &StepTimes::detect, &StepTimes::track,
                              &StepTimes::draw};
  for (size_t k = 0; k < s.steps.size(); ++k) {
    std::vector<double> ms;
    for (const auto& f : frames_) ms.push_back((f.*steps[k])[1] - (f.*steps[k])[0]);
    s.steps[k] = stat(ms);
  }
  return s;
}

}  // namespace tether
