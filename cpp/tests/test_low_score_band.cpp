// The low-score band moves with the cutoff: its floor is low_score_ratio times
// min_score. Ported from python/tests/test_low_score_band.py.

#include <gtest/gtest.h>

#include <sstream>

#include "test_util.hpp"
#include "tether/cli.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

const Box BOX = {0.0, 50.0, 20.0, 70.0};  // a still object, so every prediction is this box
constexpr double HIGH = 0.9;

// Confirm a track, then give it one detection at `score`; the track's status after it.
std::string status_after(TrackerSettings settings, double score) {
  settings.confirm_frames = 3;
  Tracker tracker(settings);
  for (int i = 0; i < 3; ++i) tracker.update({det(BOX, HIGH)});
  return only(tracker.update({det(BOX, score)})).status;
}

}  // namespace

TEST(LowScoreBand, AtTheDefaultCutoffTheFloorIsBytetracks0_1Exactly) {
  EXPECT_EQ(LOW_SCORE_RATIO * MIN_SCORE, 0.1);
}

TEST(LowScoreBand, AtTheDefaultCutoffTheBandIs0_1To0_25) {
  EXPECT_EQ(status_after({}, 0.1), "searching");
  EXPECT_EQ(status_after({}, 0.1001), "tracked");
  EXPECT_EQ(status_after({}, 0.2499), "tracked");
}

TEST(LowScoreBand, AHigherCutoffMovesTheFloorUpInProportion) {
  TrackerSettings s{.min_score = 0.5};
  EXPECT_EQ(status_after(s, 0.15), "searching");  // in the old fixed band, but under the moved floor (0.2)
  EXPECT_EQ(status_after(s, 0.2), "searching");
  EXPECT_EQ(status_after(s, 0.21), "tracked");
  EXPECT_EQ(status_after(s, 0.49), "tracked");
}

TEST(LowScoreBand, ACutoffUnder0_1StillHasABand) {
  TrackerSettings s{.min_score = 0.1};
  EXPECT_EQ(status_after(s, 0.03), "searching");
  EXPECT_EQ(status_after(s, 0.05), "tracked");
}

TEST(LowScoreBand, TheRatioCanBeSet) {
  TrackerSettings s{.min_score = 0.5, .low_score_ratio = 0.2};
  EXPECT_EQ(status_after(s, 0.15), "tracked");
  EXPECT_EQ(status_after(s, 0.1), "searching");
}

TEST(LowScoreBand, ARatioOfOneTurnsTheLowPassOff) {
  EXPECT_EQ(status_after({.min_score = 0.5, .low_score_ratio = 1.0}, 0.49), "searching");
}

TEST(LowScoreBand, TheCliSetsTheRatio) {
  auto dets = temp_path("dets.jsonl");
  auto tracks = temp_path("tracks.jsonl");
  save_detections(dets, {{det(BOX, HIGH)}, {det(BOX, HIGH)}, {det(BOX, HIGH)}, {det(BOX, 0.15)}});
  std::ostringstream out, err;
  auto status = [&](std::vector<std::string> args) {
    args.insert(args.end(), {dets.string(), tracks.string()});
    EXPECT_EQ(run_track_cli(args, out, err), 0) << err.str();
    return only(load_tracks(tracks).back()).status;
  };

  EXPECT_EQ(status({"--min-score", "0.5"}), "searching");
  EXPECT_EQ(status({"--min-score", "0.5", "--low-score-ratio", "0.2"}), "tracked");
}
