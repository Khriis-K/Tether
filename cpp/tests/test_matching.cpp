// IoU and center-distance matching (python/tests/test_matching.py).

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "tether/matching.hpp"

using tether::Box;
using tether::center_distance;
using tether::iou;
using tether::match;
using tether::match_by_distance;
using tether::Pairs;

TEST(Matching, IouOfIdenticalBoxesIsOne) { EXPECT_EQ(iou({0, 0, 10, 10}, {0, 0, 10, 10}), 1.0); }

TEST(Matching, IouOfDisjointBoxesIsZero) { EXPECT_EQ(iou({0, 0, 10, 10}, {20, 20, 30, 30}), 0.0); }

TEST(Matching, IouOfHalfOverlappingBoxes) {
  // Overlap 5x10 = 50, union 100 + 100 - 50 = 150.
  EXPECT_DOUBLE_EQ(iou({0, 0, 10, 10}, {5, 0, 15, 10}), 1.0 / 3.0);
}

TEST(Matching, IouOfZeroAreaBoxesIsZero) { EXPECT_EQ(iou({5, 5, 5, 5}, {5, 5, 5, 5}), 0.0); }

TEST(Matching, PairsEachTrackWithItsOverlappingDetection) {
  std::vector<Box> tracks = {{0, 0, 10, 10}, {100, 100, 110, 110}};
  std::vector<Box> dets = {{101, 101, 111, 111}, {1, 1, 11, 11}};

  EXPECT_EQ(match(tracks, dets), (Pairs{{0, 1}, {1, 0}}));
}

TEST(Matching, FindsTheGlobalOptimumNotGreedy) {
  // Greedy takes the single best pair (track 0, det 0, IoU ~0.82), which
  // leaves track 1 with nothing it overlaps enough. The optimal assignment
  // gives up a little on track 0 to match both.
  std::vector<Box> tracks = {{0, 0, 10, 10}, {4, 0, 14, 10}};
  std::vector<Box> dets = {{1, 0, 11, 10}, {-3, 0, 7, 10}};

  EXPECT_EQ(match(tracks, dets), (Pairs{{0, 1}, {1, 0}}));
}

TEST(Matching, KeepsEveryValidPairWhenAOnePairOptionCostsLess) {
  // T0/D0 and T1/D1 both clear MIN_IOU (~0.33 each). The alternative pairs
  // T0/D1 (~0.82) with T1/D0, which don't overlap at all. Scoring that invalid
  // pair as an ordinary cost would make the alternative look cheaper and lose
  // a valid match.
  std::vector<Box> tracks = {{0, 0, 10, 10}, {6, 0, 16, 10}};
  std::vector<Box> dets = {{-5, 0, 5, 10}, {1, 0, 11, 10}};

  EXPECT_EQ(match(tracks, dets), (Pairs{{0, 0}, {1, 1}}));
}

TEST(Matching, RejectsPairsBelowMinIou) {
  // The only possible pair barely overlaps (IoU ~0.05), so it must not match.
  EXPECT_EQ(match({{0, 0, 10, 10}}, {{9, 0, 19, 10}}), Pairs{});
}

TEST(Matching, HandlesEmptyInputs) {
  EXPECT_EQ(match({}, {{0, 0, 1, 1}}), Pairs{});
  EXPECT_EQ(match({{0, 0, 1, 1}}, {}), Pairs{});
}

TEST(Matching, RespectsAllowed) {
  EXPECT_EQ(match({{0, 0, 10, 10}}, {{0, 0, 10, 10}}, {{false}}), Pairs{});
}

TEST(Matching, CenterDistanceIsMeasuredInTrackBoxSizes) {
  // Centers (5, 5) and (35, 45): 50 px apart, track box side 10.
  EXPECT_DOUBLE_EQ(center_distance({0, 0, 10, 10}, {30, 40, 40, 50}), 5.0);
}

TEST(Matching, CenterDistanceUsesTheTrackBoxLargerSide) {
  // Track is 10 wide, 40 tall; centers 20 px apart.
  EXPECT_DOUBLE_EQ(center_distance({0, 0, 10, 40}, {20, 0, 30, 40}), 0.5);
}

TEST(Matching, CenterDistanceFromABoxWithNoSizeIsInfinite) {
  EXPECT_EQ(center_distance({5, 5, 5, 5}, {5, 5, 6, 6}), std::numeric_limits<double>::infinity());
}

TEST(Matching, ByDistancePairsBoxesThatDoNotOverlap) {
  std::vector<Box> tracks = {{0, 0, 10, 10}, {200, 0, 210, 10}};
  std::vector<Box> dets = {{230, 0, 240, 10}, {25, 0, 35, 10}};

  EXPECT_EQ(match_by_distance(tracks, dets), (Pairs{{0, 1}, {1, 0}}));
}

TEST(Matching, ByDistanceRejectsPairsBeyondMaxDistance) {
  double side = 10;
  double just_inside = tether::MAX_CENTER_DISTANCE * side - 1;
  double just_outside = tether::MAX_CENTER_DISTANCE * side + 1;

  EXPECT_EQ(match_by_distance({{0, 0, 10, 10}}, {{just_inside, 0, just_inside + 10, 10}}), (Pairs{{0, 0}}));
  EXPECT_EQ(match_by_distance({{0, 0, 10, 10}}, {{just_outside, 0, just_outside + 10, 10}}), Pairs{});
}

TEST(Matching, ByDistanceRespectsAllowed) {
  EXPECT_EQ(match_by_distance({{0, 0, 10, 10}}, {{5, 0, 15, 10}}, {{false}}), Pairs{});
}

TEST(Matching, ByDistanceAppliesEachTrackItsOwnGate) {
  // From python/tests/test_search_gate.py. Detection 0 is 2 box sizes from
  // track 0; detection 1 is 2 box sizes from track 1.
  auto box_at = [](double x) { return Box{x, 100, x + 20, 120}; };
  std::vector<Box> tracks = {box_at(0), box_at(1000)};
  std::vector<Box> dets = {box_at(40), box_at(1040), box_at(5000)};

  EXPECT_EQ(match_by_distance(tracks, dets, {}, {3.0, 1.0}), (Pairs{{0, 0}}));
  EXPECT_EQ(match_by_distance(tracks, dets, {}, {1.0, 3.0}), (Pairs{{1, 1}}));
}
