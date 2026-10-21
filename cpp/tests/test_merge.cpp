// Merging a duplicate into the searching track it came from.
// Ported from python/tests/test_merge.py. Scripted, stationary objects.

#include <gtest/gtest.h>

#include <limits>
#include <set>
#include <vector>

#include "test_util.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

constexpr double SIZE = 20.0;
constexpr double X = 400.0, Y = 300.0;
constexpr int CONFIRM = 3;
constexpr double JUMP = 6.0;  // box sizes: far outside the search gate, inside the default merge reach

Box box_at(double x, double size = SIZE) { return {x, Y, x + size, Y + size}; }

// Object A at X, confirmed, then detected only at X + offset box sizes for
// `after` frames. `others` are boxes detected on every frame.
DetectionFrames jumped(double offset, double size = SIZE, const std::vector<Box>& others = {}, int after = CONFIRM) {
  DetectionFrames frames;
  auto with_others = [&](const Box& b) {
    std::vector<Detection> dets = {det(b)};
    for (const auto& o : others) dets.push_back(det(o));
    return dets;
  };
  for (int i = 0; i < CONFIRM + 2; ++i) frames.push_back(with_others(box_at(X)));
  for (int i = 0; i < after; ++i) frames.push_back(with_others(box_at(X + offset * SIZE, size)));
  return frames;
}

TrackFrames run_merge(const DetectionFrames& frames, TrackerSettings settings = {}) {
  settings.confirm_frames = CONFIRM;
  return run(frames, settings);
}

int a_id(const TrackFrames& outputs) {
  for (const auto& t : outputs[CONFIRM + 1])
    if (t.box == box_at(X)) return t.id;
  throw std::runtime_error("A not found");
}

const Track& at_jump(const TrackFrames& outputs, double offset, double size = SIZE, int frame = -1) {
  const auto& tracks = outputs[frame < 0 ? outputs.size() + frame : frame];
  const Track* found = nullptr;
  for (const auto& t : tracks)
    if (t.box == box_at(X + offset * SIZE, size)) {
      EXPECT_EQ(found, nullptr);
      found = &t;
    }
  if (!found) throw std::runtime_error("no track at the jump");
  return *found;
}

}  // namespace

TEST(Merge, DuplicateBornWhileTheOriginalSearchesEndsWithTheOriginalId) {
  auto outputs = run_merge(jumped(JUMP));

  EXPECT_EQ(at_jump(outputs, JUMP).id, a_id(outputs));
  EXPECT_EQ(at_jump(outputs, JUMP).status, "tracked");
  EXPECT_EQ(outputs.back().size(), 1u);  // the younger ID is gone
}

TEST(Merge, NothingIsMergedBeforeTheDuplicateWouldBeConfirmed) {
  // On the duplicate's second frame it is still tentative, so A keeps searching beside it.
  auto outputs = run_merge(jumped(JUMP, SIZE, {}, CONFIRM - 1));

  const auto* a = find_id(outputs.back(), a_id(outputs));
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->status, "searching");
  EXPECT_NE(at_jump(outputs, JUMP).id, a_id(outputs));
  EXPECT_EQ(at_jump(outputs, JUMP).status, "tentative");
}

TEST(Merge, MergedTrackKeepsFollowingTheObject) {
  // The older track takes over the new track's motion; updating its own
  // coasted filter with a detection 6 box sizes away would leave it off the object.
  auto outputs = run_merge(jumped(JUMP, SIZE, {}, CONFIRM + 5));

  for (int f = -5; f < 0; ++f) EXPECT_EQ(at_jump(outputs, JUMP, SIZE, f).id, a_id(outputs)) << "frame " << f;
}

TEST(Merge, TwoObjectsSideBySideTrackedTheWholeTimeAreNeverMerged) {
  Box b = box_at(X + 3.0 * SIZE);
  DetectionFrames frames(CONFIRM + 2, {det(box_at(X))});
  for (int i = 0; i < CONFIRM + 5; ++i) frames.push_back({det(box_at(X)), det(b)});

  auto outputs = run_merge(frames);

  std::set<int> tracked;
  for (const auto& t : outputs.back())
    if (t.status == "tracked") tracked.insert(t.id);
  EXPECT_EQ(tracked.size(), 2u);
}

TEST(Merge, AMuchBiggerOrSmallerBoxIsNeverMerged) {
  for (double scale : {SEARCH_GATE_MAX_SIZE_RATIO * 1.2, 1 / (SEARCH_GATE_MAX_SIZE_RATIO * 1.2)}) {
    SCOPED_TRACE(scale);
    auto outputs = run_merge(jumped(JUMP, SIZE * scale));

    EXPECT_NE(at_jump(outputs, JUMP, SIZE * scale).id, a_id(outputs));
  }
}

TEST(Merge, ASlightlyDifferentSizeIsStillMerged) {
  double scale = SEARCH_GATE_MAX_SIZE_RATIO * 0.9;
  auto outputs = run_merge(jumped(JUMP, SIZE * scale));

  EXPECT_EQ(at_jump(outputs, JUMP, SIZE * scale).id, a_id(outputs));
}

TEST(Merge, ANearbyConfirmedTrackCapsTheReach) {
  // C sits 2 box sizes from A on the other side, so the cap is 1 box size.
  Box c = box_at(X - 2.0 * SIZE);
  for (auto [cap, merged] : {std::pair{0.5, false}, std::pair{std::numeric_limits<double>::infinity(), true}}) {
    SCOPED_TRACE(cap);
    auto outputs = run_merge(jumped(JUMP, SIZE, {c}), {.search_gate_cap = cap});

    EXPECT_EQ(at_jump(outputs, JUMP).id == a_id(outputs), merged);
  }
}

TEST(Merge, TheReachGrowsWithFramesSearched) {
  // The duplicate is confirmed on A's third missed frame (2 counted before
  // it), so the reach is 2 + 2 * growth: 4 reaches 3.5, 2 doesn't. Gate off.
  for (auto [growth, merged] : {std::pair{1.0, true}, std::pair{0.0, false}}) {
    SCOPED_TRACE(growth);
    auto outputs = run_merge(jumped(3.5), {.search_gate = 0.0,
                                           .search_gate_growth = 0.0,
                                           .merge_reach = 2.0,
                                           .merge_reach_growth = growth});

    EXPECT_EQ(at_jump(outputs, 3.5).id == a_id(outputs), merged);
  }
}

TEST(Merge, ADuplicateBeyondTheDefaultReachKeepsItsOwnId) {
  // Confirmed with 2 missed frames counted: the default reach is 10 + 2 = 12.
  auto outputs = run_merge(jumped(13.0));

  EXPECT_NE(at_jump(outputs, 13.0).id, a_id(outputs));
}

TEST(Merge, WithConfirmFramesOneNothingIsEverMerged) {
  // A track born confirmed is never tentative, so it can't be a duplicate.
  DetectionFrames frames(CONFIRM + 2, {det(box_at(X))});
  frames.push_back({det(box_at(X + JUMP * SIZE))});

  auto outputs = run(frames, {.confirm_frames = 1});

  EXPECT_EQ(outputs.back().size(), 2u);
}
