// Tracker basics and motion: scripted detections in, tracks out.
// Ported from python/tests/test_tracker.py and python/tests/test_motion.py.

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "test_util.hpp"
#include "tether/tracker.hpp"

using namespace tether;
using namespace tether::test;

namespace {

Box moving_box(int frame, double start_x = 0.0, double speed = 2.0, double y = 50.0, double size = 20.0) {
  double x = start_x + speed * frame;
  return {x, y, x + size, y + size};
}

Box box_at(double x, double y = 50.0, double size = 20.0) { return {x, y, x + size, y + size}; }

// One detection per frame, its box's left edge at each x.
TrackFrames run_xs(const std::vector<double>& xs) {
  DetectionFrames frames;
  for (double x : xs) frames.push_back({det(box_at(x))});
  return run(frames);
}

const Track& with_box(const std::vector<Track>& tracks, const Box& box) {
  for (const auto& t : tracks)
    if (t.box == box) return t;
  throw std::runtime_error("no track with that box");
}

// Each frame's single track ID is the first frame's.
void expect_one_stable_id(const TrackFrames& outputs) {
  int first = only(outputs[0]).id;
  for (size_t f = 0; f < outputs.size(); ++f) EXPECT_EQ(ids(outputs[f]), std::vector<int>{first}) << "frame " << f;
}

}  // namespace

// ---- test_tracker.py ----

TEST(Tracker, SteadilyMovingObjectKeepsTheSameIdEveryFrame) {
  DetectionFrames frames;
  for (int f = 0; f < 50; ++f) frames.push_back({det(moving_box(f))});

  expect_one_stable_id(run(frames));
}

TEST(Tracker, TrackReportsBoxClassStatusAndFramesSinceSeen) {
  auto tracks = run({{det({10.0, 20.0, 30.0, 40.0}, 0.9, 3)}})[0];

  const auto& track = only(tracks);
  EXPECT_EQ(track.box, (Box{10.0, 20.0, 30.0, 40.0}));
  EXPECT_EQ(track.cls, 3);
  EXPECT_EQ(track.status, "tentative");  // new tracks are confirmed later (test_lifecycle.cpp)
  EXPECT_EQ(track.frames_since_seen, 0);
}

TEST(Tracker, TrackBoxFollowsTheMatchedDetection) {
  DetectionFrames frames;
  for (int f = 0; f < 3; ++f) frames.push_back({det(moving_box(f))});

  auto outputs = run(frames);

  for (int f = 0; f < 3; ++f) EXPECT_EQ(outputs[f][0].box, moving_box(f));
}

TEST(Tracker, RecentPathHoldsBoxCentersOldestFirst) {
  DetectionFrames frames;
  for (int f = 0; f < 3; ++f) frames.push_back({det(moving_box(f))});

  auto track = run(frames).back()[0];

  EXPECT_EQ(track.path, (std::vector<Point>{{10.0, 60.0}, {12.0, 60.0}, {14.0, 60.0}}));
}

TEST(Tracker, RecentPathIsCapped) {
  DetectionFrames frames;
  for (int f = 0; f < 200; ++f) frames.push_back({det(moving_box(f))});

  auto track = run(frames).back()[0];

  // Only frames 170..199 remain: centers x = 2f + 10.
  ASSERT_EQ(track.path.size(), 30u);
  EXPECT_EQ(track.path.front(), (Point{350.0, 60.0}));
  EXPECT_EQ(track.path.back(), (Point{408.0, 60.0}));
}

TEST(Tracker, TwoSeparateObjectsGetDistinctStableIds) {
  DetectionFrames frames;
  for (int f = 0; f < 10; ++f) frames.push_back({det(moving_box(f, 0.0)), det(moving_box(f, 300.0))});

  auto outputs = run(frames);

  auto first_ids = ids(outputs[0]);
  EXPECT_EQ(std::set<int>(first_ids.begin(), first_ids.end()).size(), 2u);
  for (const auto& tracks : outputs) EXPECT_EQ(ids(tracks), first_ids);
}

TEST(Tracker, DetectionsOfDifferentClassesDoNotMatch) {
  auto outputs = run({{det({0.0, 0.0, 10.0, 10.0}, 0.9, 0)}, {det({0.0, 0.0, 10.0, 10.0}, 0.9, 1)}});

  EXPECT_NE(outputs[0][0].id, outputs[1][0].id);
}

TEST(Tracker, NewObjectGetsANewIdThatIsNeverReused) {
  auto outputs = run({{det({0.0, 0.0, 10.0, 10.0})}, {}, {det({0.0, 0.0, 10.0, 10.0})}});

  EXPECT_TRUE(outputs[1].empty());
  EXPECT_NE(outputs[2][0].id, outputs[0][0].id);
}

TEST(Tracker, ConfirmedTrackGetsADetectionBeforeATentativeTrackThatOverlapsItMore) {
  Box confirmed_box = {0.0, 0.0, 20.0, 20.0};
  Box tentative_box = {10.0, 0.0, 30.0, 20.0};
  Box shared = {8.0, 0.0, 28.0, 20.0};  // IoU 0.43 with the confirmed track, 0.82 with the tentative one
  DetectionFrames frames = {
      {det(confirmed_box)},
      {det(confirmed_box)},
      {det(confirmed_box), det(tentative_box)},
      {det(shared)},
  };

  auto outputs = run(frames);

  const auto& confirmed = with_box(outputs[2], confirmed_box);
  EXPECT_EQ(confirmed.status, "tracked");
  const auto& after = only(outputs[3]);
  EXPECT_EQ(after.id, confirmed.id);
  EXPECT_EQ(after.status, "tracked");
  EXPECT_EQ(after.box, shared);
}

TEST(Tracker, DetectionsBelowTheScoreThresholdAreIgnored) {
  Tracker tracker({.min_score = 0.5});

  auto tracks = tracker.update({det({0.0, 0.0, 10.0, 10.0}, 0.49), det({50.0, 50.0, 60.0, 60.0}, 0.5)});

  EXPECT_EQ(only(tracks).box, (Box{50.0, 50.0, 60.0, 60.0}));
}

TEST(Tracker, ADetectionBelowTheLowFloorDoesNotKeepATrackMatched) {
  Tracker tracker({.confirm_frames = 1, .min_score = 0.5});
  auto first = tracker.update({det(moving_box(0), 0.9)});

  // At or below LOW_SCORE a detection is dropped; between it and min_score it
  // can keep a tracked track alive (test_low_score.cpp).
  auto tracks = tracker.update({det(moving_box(1), 0.05)});

  EXPECT_EQ(only(tracks).id, first[0].id);
  EXPECT_EQ(only(tracks).status, "searching");
}

TEST(Tracker, OutputIsDeterministic) {
  // Two trackers fed the same busy scene give identical tracks every frame.
  DetectionFrames frames;
  for (int f = 0; f < 60; ++f)
    frames.push_back({det(moving_box(f, 0.0, 3.0)), det(moving_box(f, 400.0, -5.0, 52.0)),
                      det(box_at(100.0 + (f * 37 % 200), 150.0, 15.0), 0.3, f % 2),
                      det(moving_box(f, 50.0, 30.0, 300.0))});

  EXPECT_EQ(run(frames), run(frames));
}

// ---- test_motion.py ----

TEST(Motion, ObjectJumpingFarEveryFrameKeepsTheSameId) {
  // Speeds up smoothly from 2 to 60 px/frame, then holds 60. Once the step
  // passes the box size (20) the boxes stop overlapping between frames, so
  // only the predicted box can match.
  std::vector<double> xs;
  double x = 0.0, speed = 2.0;
  for (int i = 0; i < 50; ++i) {
    xs.push_back(x);
    x += speed;
    speed = std::min(speed + 4.0, 60.0);
  }
  ASSERT_EQ(xs[49] - xs[48], 60.0);

  expect_one_stable_id(run_xs(xs));
}

TEST(Motion, ObjectAlreadyFastOnItsFirstFrameKeepsTheSameId) {
  // 50 px/frame from the very first frame: no overlap, no velocity history.
  std::vector<double> xs;
  for (int f = 0; f < 20; ++f) xs.push_back(50.0 * f);

  expect_one_stable_id(run_xs(xs));
}

TEST(Motion, FastObjectsMovingTowardEachOtherKeepTheirIds) {
  DetectionFrames frames;
  for (int f = 0; f < 12; ++f) frames.push_back({det(box_at(30.0 * f, 0.0)), det(box_at(700.0 - 30.0 * f, 200.0))});

  auto outputs = run(frames);

  auto id_in_lane = [](const std::vector<Track>& tracks, double y) {
    for (const auto& t : tracks)
      if (t.box[1] == y) return t.id;
    return -1;
  };
  int top = id_in_lane(outputs[0], 0.0), bottom = id_in_lane(outputs[0], 200.0);
  EXPECT_NE(top, bottom);
  for (const auto& tracks : outputs) {
    EXPECT_EQ(tracks.size(), 2u);
    EXPECT_EQ(id_in_lane(tracks, 0.0), top);
    EXPECT_EQ(id_in_lane(tracks, 200.0), bottom);
  }
}

TEST(Motion, FastObjectsCrossingPathsDoNotSwapIds) {
  // Same lane (4 px apart in y), 30 px/frame each way; they pass through each
  // other around frame 10, so mid-crossing both detections overlap both predictions.
  auto rightward = [](int f) { return box_at(30.0 * f, 50.0); };
  auto leftward = [](int f) { return box_at(600.0 - 30.0 * f, 54.0); };
  DetectionFrames frames;
  for (int f = 0; f < 20; ++f) frames.push_back({det(rightward(f)), det(leftward(f))});

  auto outputs = run(frames);

  int right_id = with_box(outputs[0], rightward(0)).id;
  int left_id = with_box(outputs[0], leftward(0)).id;
  EXPECT_NE(right_id, left_id);
  for (int f = 0; f < 20; ++f) {
    EXPECT_EQ(with_box(outputs[f], rightward(f)).id, right_id) << "frame " << f;
    EXPECT_EQ(with_box(outputs[f], leftward(f)).id, left_id) << "frame " << f;
  }
}

TEST(Motion, ObjectWhoseBoxShrinksSharplyKeepsItsId) {
  // 40 px box, then a 12 px box at the same center (e.g. the detector now sees
  // only part of it). One shrink is not a shrink rate: the predicted box must
  // stay a real box near the object.
  auto centered = [](double size) { return Box{100 - size / 2, 100 - size / 2, 100 + size / 2, 100 + size / 2}; };
  DetectionFrames frames = {{det(centered(40.0))}};
  for (int i = 0; i < 5; ++i) frames.push_back({det(centered(12.0))});

  auto outputs = run(frames);

  expect_one_stable_id(outputs);
  for (const auto& tracks : outputs) {
    const auto& p = tracks[0].predicted_box;
    EXPECT_GT(p[2], p[0]);
    EXPECT_GT(p[3], p[1]);
  }
}

TEST(Motion, FarAwayDetectionAfterANewTrackGetsANewId) {
  auto outputs = run_xs({0.0, 500.0});

  EXPECT_NE(outputs[1][0].id, outputs[0][0].id);
}

TEST(Motion, EstablishedTrackDoesNotTakeANearbyNonOverlappingDetection) {
  // A still object seen for 5 frames, then the only detection is 3 box sizes
  // away. The track's prediction says "still here", so that's someone else.
  auto outputs = run_xs({0.0, 0.0, 0.0, 0.0, 0.0, 60.0});

  EXPECT_NE(with_box(outputs[5], box_at(60.0)).id, outputs[0][0].id);
}

TEST(Motion, NewTrackPredictsItStaysWhereItIs) {
  auto track = run_xs({10.0})[0][0];

  EXPECT_EQ(track.predicted_box, track.box);
}

TEST(Motion, PredictedBoxLeadsASteadilyMovingObjectToItsNextBox) {
  std::vector<double> xs;
  for (int f = 0; f < 20; ++f) xs.push_back(3.0 * f);

  auto track = run_xs(xs).back()[0];

  EXPECT_TRUE(boxes_near(track.predicted_box, box_at(3.0 * 20), 0.5));
}
