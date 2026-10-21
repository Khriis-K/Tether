// The tracker: one frame of detections in, the current tracks out.
// A port of python/tether/tracker.py; see there for why each stage exists.
#pragma once

#include <deque>
#include <functional>
#include <vector>

#include "tether/formats.hpp"
#include "tether/kalman.hpp"
#include "tether/matching.hpp"

namespace tether {

inline constexpr int PATH_LENGTH = 30;          // box centers kept per track for trails
inline constexpr int CONFIRM_FRAMES = 3;        // matched frames in a row before a new track is confirmed
inline constexpr int MAX_MISSING_FRAMES = 30;   // missed frames in a row a confirmed track survives
inline constexpr double MIN_SCORE = 0.25;       // lowest detector confidence the tracker uses
inline constexpr double LOW_SCORE = 0.1;        // low detections score above this and below min_score
inline constexpr double LOW_MIN_IOU = 0.5;      // pairs with a low detection that overlap less never match
inline constexpr double SEARCH_GATE = 1.5;      // after one missed frame the gate is this + growth
inline constexpr double SEARCH_GATE_GROWTH = 0.25;
inline constexpr double SEARCH_GATE_CAP = 0.5;  // times the distance to the nearest other confirmed track
inline constexpr double SEARCH_GATE_MAX_SIZE_RATIO = 1.5;
inline constexpr double MERGE_REACH = 10.0;     // after one missed frame the reach is this + growth
inline constexpr double MERGE_REACH_GROWTH = 1.0;
inline constexpr double COAST_DAMPING = 1.0;    // 1 is undamped, 0 holds still

struct TrackerSettings {
  int confirm_frames = CONFIRM_FRAMES;
  int max_missing_frames = MAX_MISSING_FRAMES;
  double min_score = MIN_SCORE;
  double search_gate = SEARCH_GATE;  // this and search_gate_growth both 0 turns the gate off
  double search_gate_growth = SEARCH_GATE_GROWTH;
  double search_gate_cap = SEARCH_GATE_CAP;  // infinity is no cap
  double coast_damping = COAST_DAMPING;
  double merge_reach = MERGE_REACH;  // this and merge_reach_growth both 0 turns merging off
  double merge_reach_growth = MERGE_REACH_GROWTH;
};

class Tracker {
 public:
  explicit Tracker(TrackerSettings settings = {});

  // Advance one frame. Returns the live tracks, sorted by ID.
  std::vector<Track> update(const std::vector<Detection>& detections);

 private:
  struct State {
    int id;
    Box box;  // the last matched detection
    int cls;
    BoxFilter motion;
    std::deque<Point> path;
    int hits = 0;  // detections matched so far, counting the first
    bool confirmed = false;
    int frames_since_seen = 0;

    void observe(const Box& b, int confirm_frames);
    void take_over(const State& other);
    const char* status() const;
    Track snapshot() const;
  };

  // Pairs tracks with detection boxes, given which pairs are allowed; returns
  // (track, detection) indices into its arguments.
  using Matcher = std::function<Pairs(const std::vector<const State*>&, const std::vector<Box>&, const Mask&)>;

  // Pairs the `wanted` tracks not yet in `pairs` with the detections not yet in `pairs`.
  Pairs match_unmatched(const std::function<bool(const State&)>& wanted, const Matcher& matcher,
                        const std::vector<Detection>& detections, const Mask& allowed, const Pairs& pairs) const;
  Pairs by_search_gate(const std::vector<const State*>& tracks, const std::vector<Box>& boxes,
                       const Mask& allowed) const;
  Pairs merge_duplicates(const Pairs& pairs, const std::vector<Detection>& detections);
  bool is_duplicate_of(const State& old, int cls, const Box& box) const;
  double gate(const State& t) const;
  double reach(const State& t) const;
  double capped(const State& t, double distance) const;

  TrackerSettings settings_;
  std::vector<State> tracks_;  // sorted by ID
  int next_id_ = 1;
};

}  // namespace tether
