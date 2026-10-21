// Velocity damping while a track coasts. Ported from python/tests/test_coast_damping.py.

#include <gtest/gtest.h>

#include <vector>

#include "test_util.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

constexpr double SIZE = 20.0;
constexpr double SPEED = 10.0;
constexpr int CONFIRM = 3;
constexpr int MATCHED = 10, MISSED = 5;

Detection det_at(int frame) {
  double x = SPEED * frame;
  return det({x, 0.0, x + SIZE, SIZE});
}

DetectionFrames moving_then_missing(int matched = MATCHED, int missed = MISSED) {
  DetectionFrames frames;
  for (int f = 0; f < matched; ++f) frames.push_back({det_at(f)});
  for (int i = 0; i < missed; ++i) frames.push_back({});
  return frames;
}

TrackFrames run_damped(const DetectionFrames& frames, double damping, double search_gate = SEARCH_GATE) {
  return run(frames, {.confirm_frames = CONFIRM, .search_gate = search_gate, .coast_damping = damping});
}

std::vector<double> predicted_x(const TrackFrames& outputs) {
  std::vector<double> xs;
  for (const auto& tracks : outputs) xs.push_back(only(tracks).predicted_box[0]);
  return xs;
}

// Steps between predictions from the last matched frame on.
std::vector<double> coasted_steps(const TrackFrames& outputs) {
  auto xs = predicted_x(outputs);
  std::vector<double> steps;
  for (size_t i = MATCHED; i < xs.size(); ++i) steps.push_back(xs[i] - xs[i - 1]);
  return steps;
}

}  // namespace

TEST(CoastDamping, VelocityShrinksByTheDampingEachCoastedFrame) {
  for (double damping : {0.5, 0.8}) {
    SCOPED_TRACE(damping);
    auto coasted = coasted_steps(run_damped(moving_then_missing(), damping));

    ASSERT_EQ(coasted.size(), static_cast<size_t>(MISSED));
    EXPECT_GT(coasted[0], 0);
    for (size_t i = 1; i < coasted.size(); ++i) EXPECT_NEAR(coasted[i], damping * coasted[i - 1], 1e-6 * coasted[i - 1]);
  }
}

TEST(CoastDamping, FirstCoastedStepIsDampedToo) {
  auto undamped = coasted_steps(run_damped(moving_then_missing(), 1.0));
  auto damped = coasted_steps(run_damped(moving_then_missing(), 0.5));

  EXPECT_NEAR(damped[0], 0.5 * undamped[0], 1e-6 * undamped[0]);
}

TEST(CoastDamping, ZeroDampingHoldsTheCoastedPositionStill) {
  auto xs = predicted_x(run_damped(moving_then_missing(), 0.0));

  for (size_t i = MATCHED; i < xs.size(); ++i) EXPECT_EQ(xs[i], xs[MATCHED - 1]);
}

TEST(CoastDamping, MatchedFramesAreNotDamped) {
  auto frames = moving_then_missing(MATCHED, 0);

  EXPECT_EQ(run_damped(frames, 0.5), run_damped(frames, 1.0));
}

TEST(CoastDamping, VelocityRecoversOnceTheTrackMatchesAgain) {
  // Missed for a few frames, then matched again (by the search gate, as the
  // damped prediction lags the object) for many frames at the same speed.
  int gap = 4;
  DetectionFrames frames;
  for (int f = 0; f < MATCHED; ++f) frames.push_back({det_at(f)});
  for (int i = 0; i < gap; ++i) frames.push_back({});
  for (int f = MATCHED + gap; f < MATCHED + gap + 30; ++f) frames.push_back({det_at(f)});

  auto outputs = run_damped(frames, 0.5, 5.0);

  for (const auto& tracks : outputs) EXPECT_EQ(only(tracks).id, outputs[0][0].id);
  // Damping that stuck would leave the prediction lagging behind the object;
  // undamped, it lands where the object is on the next frame.
  EXPECT_NEAR(predicted_x(outputs).back(), SPEED * frames.size(), 0.5);
}

TEST(CoastDamping, DefaultIsUndamped) {
  auto frames = moving_then_missing();

  EXPECT_EQ(run(frames, {.confirm_frames = CONFIRM}), run_damped(frames, 1.0));
}
