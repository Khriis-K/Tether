// Re-matching a searching track by center distance: the growing gate, its cap
// by the nearest other confirmed track, and the size check. Ported from
// python/tests/test_search_gate.py, test_search_gate_cap.py and
// test_search_gate_size.py. Scripted, stationary objects unless noted.

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/matching.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

constexpr double SIZE = 20.0;
constexpr int CONFIRM = 3;
constexpr double INF = std::numeric_limits<double>::infinity();

// The track at the frame before last that is searching, and whether the track
// with `returned` box, matched on the last frame, has its ID.
bool kept_its_id(const TrackFrames& outputs, const Box& returned) {
  const Track* searching = nullptr;
  for (const auto& t : outputs[outputs.size() - 2])
    if (t.status == "searching") {
      EXPECT_EQ(searching, nullptr) << "more than one searching track";
      searching = &t;
    }
  if (!searching) throw std::runtime_error("no searching track");
  const Track* at_returned = nullptr;
  for (const auto& t : outputs.back())
    if (t.box == returned && t.frames_since_seen == 0) {
      EXPECT_EQ(at_returned, nullptr) << "more than one track at the returned box";
      at_returned = &t;
    }
  if (!at_returned) throw std::runtime_error("no track at the returned box");
  return at_returned->id == searching->id;
}

const Track& by_box(const std::vector<Track>& tracks, const Box& box) {
  const Track* found = nullptr;
  for (const auto& t : tracks)
    if (t.box == box && t.frames_since_seen == 0) {
      EXPECT_EQ(found, nullptr);
      found = &t;
    }
  if (!found) throw std::runtime_error("no track at that box");
  return *found;
}

// A stationary object at `box` confirmed over a few frames, then undetected for `missed` frames.
DetectionFrames confirmed_then_missing(int missed, const Box& box) {
  DetectionFrames frames(CONFIRM + 2, {det(box)});
  frames.resize(frames.size() + missed);
  return frames;
}

DetectionFrames plus(DetectionFrames frames, const std::vector<Detection>& last) {
  frames.push_back(last);
  return frames;
}

// ---- test_search_gate.py: the gate and its growth ----

namespace gate_test {

constexpr double X = 100.0, Y = 100.0;
constexpr double SEARCH_GATE = 1.0, SEARCH_GATE_GROWTH = 0.5;  // test values, not the tracker's defaults

Box box_at(double x) { return {x, Y, x + SIZE, Y + SIZE}; }
double gate(int frames_since_seen) { return SEARCH_GATE + SEARCH_GATE_GROWTH * frames_since_seen; }

TrackFrames run_gate(const DetectionFrames& frames, double search_gate = SEARCH_GATE,
                     double growth = SEARCH_GATE_GROWTH) {
  return run(frames, {.confirm_frames = CONFIRM, .search_gate = search_gate, .search_gate_growth = growth});
}

// A narrow box with the same center and height: IoU below MIN_IOU, but
// distance 0 and the same size, so any gate above 0 takes it.
const Box CENTERED = {X + 8.0, Y, X + 12.0, Y + SIZE};

}  // namespace gate_test

}  // namespace

TEST(SearchGate, SearchingTrackRematchesADetectionOutsideIouRangeButInsideTheGate) {
  using namespace gate_test;
  double offset = std::max(1.25, gate(1) - 0.25);  // box sizes: at least 1.25 so the boxes don't overlap
  ASSERT_LT(offset, gate(1));
  Box returned = box_at(X + offset * SIZE);
  ASSERT_LT(iou(box_at(X), returned), MIN_IOU);

  auto outputs = run_gate(plus(confirmed_then_missing(1, box_at(X)), {det(returned)}));

  const auto& searching = only(outputs[outputs.size() - 2]);
  EXPECT_EQ(searching.status, "searching");
  EXPECT_EQ(searching.frames_since_seen, 1);
  const auto& track = only(outputs.back());
  EXPECT_EQ(track.id, searching.id);
  EXPECT_EQ(track.status, "tracked");
  EXPECT_EQ(track.box, returned);
}

TEST(SearchGate, GateWidensWithFramesSinceSeen) {
  using namespace gate_test;
  // Outside the gate after one missed frame, inside it after five.
  double offset = (gate(1) + gate(5)) / 2;
  ASSERT_TRUE(gate(1) < offset && offset < gate(5));
  Box returned = box_at(X + offset * SIZE);

  auto after_one = run_gate(plus(confirmed_then_missing(1, box_at(X)), {det(returned)}));
  auto after_five = run_gate(plus(confirmed_then_missing(5, box_at(X)), {det(returned)}));

  EXPECT_NE(by_box(after_one.back(), returned).id, only(after_one[after_one.size() - 2]).id);
  EXPECT_EQ(only(after_five.back()).id, only(after_five[after_five.size() - 2]).id);
  EXPECT_EQ(only(after_five.back()).status, "tracked");
}

TEST(SearchGate, DetectionOutsideTheGateStartsANewTrack) {
  using namespace gate_test;
  Box returned = box_at(X + (gate(1) + 0.25) * SIZE);

  auto outputs = run_gate(plus(confirmed_then_missing(1, box_at(X)), {det(returned)}));

  const auto& old = only(outputs[outputs.size() - 2]);
  const auto* still = find_id(outputs.back(), old.id);
  ASSERT_NE(still, nullptr);
  EXPECT_EQ(still->status, "searching");
  const auto& fresh = by_box(outputs.back(), returned);
  EXPECT_NE(fresh.id, old.id);
  EXPECT_EQ(fresh.status, "tentative");
}

TEST(SearchGate, SearchingTrackTakesItsDetectionBackFromTheTentativeTrackItStarted) {
  using namespace gate_test;
  // The failure on Anti-UAV: a track misses one jump, the detection starts a
  // tentative track there, and on the next frame that tentative track overlaps
  // the detection fully. The searching track should still get it.
  double offset = std::max(1.25, std::min(gate(0), gate(1)) - 0.25);
  ASSERT_LT(offset, gate(1));
  Box jumped = box_at(X + offset * SIZE);
  ASSERT_LT(iou(box_at(X), jumped), MIN_IOU);
  DetectionFrames frames(CONFIRM + 2, {det(box_at(X))});
  frames.push_back({det(jumped)});
  frames.push_back({det(jumped)});

  auto outputs = run_gate(frames);

  int original = only(outputs[CONFIRM + 1]).id;
  EXPECT_EQ(by_box(outputs.back(), jumped).id, original);
  EXPECT_EQ(by_box(outputs.back(), jumped).status, "tracked");
}

TEST(SearchGate, GateIsOnByDefault) {
  using namespace gate_test;
  ASSERT_LT(iou(box_at(X), CENTERED), MIN_IOU);

  auto outputs = run(plus(confirmed_then_missing(1, box_at(X)), {det(CENTERED)}), {.confirm_frames = CONFIRM});

  EXPECT_EQ(only(outputs.back()).id, only(outputs[outputs.size() - 2]).id);
  EXPECT_EQ(only(outputs.back()).status, "tracked");
}

TEST(SearchGate, ZeroGateNeverMatchesEvenADetectionCenteredOnThePrediction) {
  using namespace gate_test;
  ASSERT_LT(iou(box_at(X), CENTERED), MIN_IOU);

  auto outputs = run_gate(plus(confirmed_then_missing(1, box_at(X)), {det(CENTERED)}), 0.0, 0.0);

  EXPECT_NE(by_box(outputs.back(), CENTERED).id, only(outputs[outputs.size() - 2]).id);
}

// ---- test_search_gate_cap.py: capped by the nearest other confirmed track ----

namespace {
namespace cap_test {

constexpr double X = 200.0, Y = 100.0;
constexpr double SEARCH_GATE = 1.0, SEARCH_GATE_GROWTH = 0.5;  // gate after one missed frame: 1.5 box sizes
constexpr double CAP = 0.5;
constexpr double RETURN_OFFSET = 1.25;  // box sizes: inside the full gate, and no overlap with the old box

Box box_at(double x) { return {x, Y, x + SIZE, Y + SIZE}; }

TrackFrames run_cap(const DetectionFrames& frames, double cap = CAP) {
  return run(frames, {.confirm_frames = CONFIRM,
                      .search_gate = SEARCH_GATE,
                      .search_gate_growth = SEARCH_GATE_GROWTH,
                      .search_gate_cap = cap});
}

// Object A at X, confirmed then missed for one frame and back at `returned`.
// `neighbor` is an object B detected on every frame, if any.
DetectionFrames searching_next_to(const std::vector<Box>& neighbor, const Box& returned) {
  std::vector<Detection> others;
  for (const auto& b : neighbor) others.push_back(det(b));
  DetectionFrames frames;
  for (int i = 0; i < CONFIRM + 2; ++i) {
    frames.push_back({det(box_at(X))});
    frames.back().insert(frames.back().end(), others.begin(), others.end());
  }
  frames.push_back(others);
  frames.push_back({det(returned)});
  frames.back().insert(frames.back().end(), others.begin(), others.end());
  return frames;
}

Box returned_at(double offset) {
  Box returned = box_at(X + offset * SIZE);
  if (offset >= 1.0) EXPECT_LT(iou(box_at(X), returned), MIN_IOU);
  return returned;
}

}  // namespace cap_test
}  // namespace

TEST(SearchGateCap, IsolatedSearchingTrackRematchesWithinItsFullGate) {
  using namespace cap_test;
  Box returned = returned_at(RETURN_OFFSET);

  EXPECT_TRUE(kept_its_id(run_cap(searching_next_to({}, returned)), returned));
}

TEST(SearchGateCap, NearbyTrackShrinksTheGateToTheCap) {
  using namespace cap_test;
  // B is 2 box sizes away on the other side, so the cap is 1.0: the detection
  // at 1.25 is beyond it, though inside the full gate of 1.5.
  Box neighbor = box_at(X - 2.0 * SIZE);
  Box returned = returned_at(RETURN_OFFSET);
  ASSERT_TRUE(CAP * 2.0 < RETURN_OFFSET && RETURN_OFFSET < cap_test::SEARCH_GATE + cap_test::SEARCH_GATE_GROWTH);

  auto outputs = run_cap(searching_next_to({neighbor}, returned));

  EXPECT_FALSE(kept_its_id(outputs, returned));
  EXPECT_EQ(by_box(outputs.back(), returned).status, "tentative");
}

TEST(SearchGateCap, WithoutTheCapTheSameDetectionRematches) {
  using namespace cap_test;
  Box returned = returned_at(RETURN_OFFSET);

  EXPECT_TRUE(kept_its_id(run_cap(searching_next_to({box_at(X - 2.0 * SIZE)}, returned), INF), returned));
}

TEST(SearchGateCap, DetectionInsideTheCapStillRematches) {
  using namespace cap_test;
  Box returned = returned_at(0.75);  // under the cap of 1.0, and too little overlap for IoU
  ASSERT_LT(iou(box_at(X), returned), MIN_IOU);

  EXPECT_TRUE(kept_its_id(run_cap(searching_next_to({box_at(X - 2.0 * SIZE)}, returned)), returned));
}

TEST(SearchGateCap, CapMeasuresToOtherTracksNeverTheTrackItself) {
  using namespace cap_test;
  // B is 3 box sizes away, so the cap is 1.5 and doesn't bind. Measured to
  // itself the distance would be 0, and the track could never re-match.
  Box returned = returned_at(RETURN_OFFSET);

  EXPECT_TRUE(kept_its_id(run_cap(searching_next_to({box_at(X - 3.0 * SIZE)}, returned)), returned));
}

TEST(SearchGateCap, TentativeTracksDontCapTheGate) {
  using namespace cap_test;
  // A misses a jump, the detection starts a tentative track there, and on the
  // next frame A should still take its detection back from it.
  Box jumped = returned_at(RETURN_OFFSET);
  DetectionFrames frames(CONFIRM + 2, {det(box_at(X))});
  frames.push_back({det(jumped)});
  frames.push_back({det(jumped)});

  auto outputs = run_cap(frames);

  int original = outputs[CONFIRM + 1][0].id;
  const auto& at_jumped = by_box(outputs.back(), jumped);
  EXPECT_EQ(at_jumped.id, original);
  EXPECT_EQ(at_jumped.status, "tracked");
}

TEST(SearchGateCap, CapIsOnByDefault) {
  using namespace cap_test;
  // The default gate after one missed frame (1.75) reaches the detection at
  // 1.25; the default cap (half the 2 box sizes to B) doesn't.
  Box returned = returned_at(RETURN_OFFSET);

  EXPECT_FALSE(kept_its_id(run(searching_next_to({box_at(X - 2.0 * SIZE)}, returned)), returned));
}

// ---- test_search_gate_size.py: a much bigger or smaller box is someone else ----

namespace {
namespace size_test {

constexpr double X = 200.0, Y = 100.0;

Box box_at(double cx, double size = SIZE) {
  double cy = Y + SIZE / 2;
  return {cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2};
}

TrackFrames run_size(const DetectionFrames& frames) { return run(frames, {.confirm_frames = CONFIRM}); }

}  // namespace size_test
}  // namespace

TEST(SearchGateSize, DetectionMuchBiggerOrSmallerStartsANewTrack) {
  using namespace size_test;
  for (double scale : {2.0, 0.5}) {
    SCOPED_TRACE(scale);
    // Same center, so well inside the gate, but not the same object.
    ASSERT_TRUE(scale > SEARCH_GATE_MAX_SIZE_RATIO || 1 / scale > SEARCH_GATE_MAX_SIZE_RATIO);
    Box old = box_at(X);
    Box returned = box_at(X, SIZE * scale);
    ASSERT_LT(iou(old, returned), MIN_IOU);

    EXPECT_FALSE(kept_its_id(run_size(plus(confirmed_then_missing(1, old), {det(returned)})), returned));
  }
}

TEST(SearchGateSize, ModestSizeChangeStillRematches) {
  using namespace size_test;
  Box old = box_at(X);
  Box returned = box_at(X + 1.25 * SIZE, SIZE * 1.25);  // no overlap, inside the gate
  ASSERT_LT(1.25, SEARCH_GATE_MAX_SIZE_RATIO);
  ASSERT_LT(iou(old, returned), MIN_IOU);

  EXPECT_TRUE(kept_its_id(run_size(plus(confirmed_then_missing(1, old), {det(returned)})), returned));
}

TEST(SearchGateSize, SizeIsComparedToTheLastObservedBoxNotTheCoastedPrediction) {
  using namespace size_test;
  // An object growing every frame: coasting, its predicted box keeps growing,
  // so by the time it's seen again at its last size the prediction is far bigger.
  DetectionFrames frames;
  for (int i = 0; i < 25; ++i) frames.push_back({det(box_at(X, SIZE + 4.0 * i))});
  double last_size = SIZE + 4.0 * 24;
  Box returned = box_at(X + last_size / 2, last_size);  // half a box over, so IoU can't match it
  for (int i = 0; i < 29; ++i) frames.push_back({});
  frames.push_back({det(returned)});

  auto outputs = run_size(frames);

  const auto& predicted = outputs[outputs.size() - 2][0].predicted_box;
  EXPECT_GT((predicted[3] - predicted[1]) / last_size, SEARCH_GATE_MAX_SIZE_RATIO);
  EXPECT_LT(iou(predicted, returned), MIN_IOU);
  EXPECT_TRUE(kept_its_id(outputs, returned));
}
