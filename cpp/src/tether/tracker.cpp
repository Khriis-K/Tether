// A port of python/tether/tracker.py. Its module docstring explains each
// matching stage, merging, and the lifecycle; the order of every loop here
// follows the Python so the two trackers break ties the same way.
//
// The Python's single-target prior (#34) is a diagnostic upper bound that is
// never ported.

#include "tether/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace tether {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

Point center(const Box& b) { return {(b[0] + b[2]) / 2, (b[1] + b[3]) / 2}; }

// Appends to a trail, keeping only the last PATH_LENGTH points.
void push_capped(std::deque<Point>& path, const Point& p) {
  path.push_back(p);
  if (path.size() > static_cast<size_t>(PATH_LENGTH)) path.pop_front();
}

// Heights within SEARCH_GATE_MAX_SIZE_RATIO of each other; a zero height matches nothing.
bool same_size(const Box& last, const Box& det) {
  double a = last[3] - last[1], b = det[3] - det[1];
  return a > 0 && b > 0 && std::max(a / b, b / a) <= SEARCH_GATE_MAX_SIZE_RATIO;
}

template <typename Tracks>
std::vector<Box> predicted_boxes(const Tracks& tracks) {
  std::vector<Box> boxes;
  for (const auto* t : tracks) boxes.push_back(t->motion.box());
  return boxes;
}

}  // namespace

void Tracker::State::observe(const Box& b, int confirm_frames) {
  box = b;
  hits += 1;
  frames_since_seen = 0;
  confirmed = confirmed || hits >= confirm_frames;
  push_capped(path, center(b));
}

void Tracker::State::take_over(const State& other) {
  motion = other.motion;
  hits += other.hits;
  for (const auto& p : other.path) push_capped(path, p);
}

const char* Tracker::State::status() const {
  if (!confirmed) return "tentative";
  return frames_since_seen == 0 ? "tracked" : "searching";
}

Track Tracker::State::snapshot() const {
  return {id, box, cls, status(), motion.box(), frames_since_seen, {path.begin(), path.end()}};
}

Tracker::Tracker(TrackerSettings settings) : settings_(settings) {}

std::vector<Track> Tracker::update(const std::vector<Detection>& input) {
  // High detections come first, so indices below high_count are high ones.
  std::vector<Detection> detections;
  for (const auto& d : input)
    if (d.score >= settings_.min_score) detections.push_back(d);
  size_t high_count = detections.size();
  double low_score = settings_.low_score_ratio * settings_.min_score;
  for (const auto& d : input)
    if (low_score < d.score && d.score < settings_.min_score) detections.push_back(d);

  // Every track's filter was stepped forward at the end of the last update, so
  // its box is already the prediction for this frame.
  Mask high_allowed(tracks_.size(), std::vector<bool>(detections.size()));
  Mask low_allowed = high_allowed;
  for (size_t ti = 0; ti < tracks_.size(); ++ti)
    for (size_t di = 0; di < detections.size(); ++di) {
      bool same_class = tracks_[ti].cls == detections[di].cls;
      (di < high_count ? high_allowed : low_allowed)[ti][di] = same_class;
    }

  auto by_iou = [](double min_iou) -> Matcher {
    return [min_iou](const std::vector<const State*>& tracks, const std::vector<Box>& boxes, const Mask& allowed) {
      return match(predicted_boxes(tracks), boxes, allowed, min_iou);
    };
  };
  auto by_one_hit_distance = [](const std::vector<const State*>& tracks, const std::vector<Box>& boxes,
                                const Mask& allowed) { return match_by_distance(predicted_boxes(tracks), boxes, allowed); };
  auto append = [](Pairs& pairs, const Pairs& more) { pairs.insert(pairs.end(), more.begin(), more.end()); };

  // 1. Confirmed tracks against the high detections, by IoU.
  Pairs pairs = match_unmatched([](const State& t) { return t.confirmed; }, by_iou(MIN_IOU), detections,
                                high_allowed, {});
  // 2. Tracked tracks still unmatched against the low detections.
  append(pairs, match_unmatched([](const State& t) { return t.confirmed && t.frames_since_seen == 0; },
                                by_iou(LOW_MIN_IOU), detections, low_allowed, pairs));
  // 3. Searching tracks against the high detections left, by center distance within the gate.
  append(pairs, match_unmatched([this](const State& t) { return t.confirmed && t.frames_since_seen > 0 && gate(t) > 0; },
                                std::bind_front(&Tracker::by_search_gate, this), detections, high_allowed, pairs));
  // 4. Tentative tracks against the high detections left, by IoU.
  append(pairs, match_unmatched([](const State& t) { return !t.confirmed; }, by_iou(MIN_IOU), detections,
                                high_allowed, pairs));
  // 5. Tracks seen once (no velocity yet) get a second chance by center distance.
  append(pairs, match_unmatched([](const State& t) { return t.hits == 1; }, by_one_hit_distance, detections,
                                high_allowed, pairs));
  pairs = merge_duplicates(pairs, detections);

  std::vector<State> survivors;
  std::vector<bool> matched_track(tracks_.size(), false);
  std::vector<bool> matched_det(detections.size(), false);
  for (auto [ti, di] : pairs) {
    auto& track = tracks_[ti];
    track.motion.update(detections[di].box);
    track.observe(detections[di].box, settings_.confirm_frames);
    matched_track[ti] = true;
    matched_det[di] = true;
  }
  for (auto [ti, di] : pairs) survivors.push_back(std::move(tracks_[ti]));
  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    if (matched_track[ti]) continue;
    auto& track = tracks_[ti];
    track.frames_since_seen += 1;
    if (track.confirmed && track.frames_since_seen <= settings_.max_missing_frames)
      survivors.push_back(std::move(track));
  }
  for (size_t di = 0; di < high_count; ++di) {
    if (matched_det[di]) continue;
    const auto& box = detections[di].box;
    State track{next_id_++, box, detections[di].cls, BoxFilter(box), {}};
    track.observe(box, settings_.confirm_frames);
    survivors.push_back(std::move(track));
  }

  for (auto& track : survivors) track.motion.predict(track.frames_since_seen > 0 ? settings_.coast_damping : 1.0);
  std::sort(survivors.begin(), survivors.end(), [](const State& a, const State& b) { return a.id < b.id; });
  tracks_ = std::move(survivors);

  std::vector<Track> out;
  for (const auto& t : tracks_) out.push_back(t.snapshot());
  return out;
}

double Tracker::gate(const State& t) const {
  // frames_since_seen still counts up to the last frame: this one isn't decided yet.
  return settings_.search_gate + settings_.search_gate_growth * t.frames_since_seen;
}

double Tracker::reach(const State& t) const {
  // Counted the same way as the gate's.
  return settings_.merge_reach + settings_.merge_reach_growth * t.frames_since_seen;
}

double Tracker::capped(const State& t, double distance) const {
  // `distance`, at most search_gate_cap times the distance to the nearest other confirmed track.
  if (settings_.search_gate_cap == INF) return distance;  // no cap: skip the scan over every track
  double nearest = INF;
  for (const auto& other : tracks_)
    if (&other != &t && other.confirmed) nearest = std::min(nearest, center_distance(t.motion.box(), other.motion.box()));
  return std::min(distance, settings_.search_gate_cap * nearest);
}

Pairs Tracker::merge_duplicates(const Pairs& pairs, const std::vector<Detection>& detections) {
  // Hands each pair that would confirm a tentative track to an older track it
  // duplicates. Runs before any pair is applied, so every box is still this
  // frame's prediction and the tentative tracks don't count toward the cap.
  std::set<int> unmatched;
  for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) unmatched.insert(ti);
  for (auto [ti, di] : pairs) unmatched.erase(ti);

  Pairs merged;
  for (auto [ti, di] : pairs) {
    const auto& fresh = tracks_[ti];
    const auto& box = detections[di].box;
    if (!fresh.confirmed && fresh.hits + 1 >= settings_.confirm_frames) {
      // The nearest duplicate, by (distance, index) as Python's min over tuples.
      std::tuple<double, int> best{INF, -1};
      for (int oi : unmatched)
        if (is_duplicate_of(tracks_[oi], fresh.cls, box))
          best = std::min(best, std::tuple{center_distance(tracks_[oi].motion.box(), box), oi});
      if (int oi = std::get<1>(best); oi >= 0) {
        unmatched.erase(oi);
        tracks_[oi].take_over(fresh);
        ti = oi;
      }
    }
    merged.emplace_back(ti, di);
  }
  return merged;
}

bool Tracker::is_duplicate_of(const State& old, int cls, const Box& box) const {
  // Whether a new track confirmed on `box` is `old`'s object under a younger ID.
  return old.confirmed && old.cls == cls && same_size(old.box, box) &&
         center_distance(old.motion.box(), box) <= capped(old, reach(old));
}

Pairs Tracker::by_search_gate(const std::vector<const State*>& tracks, const std::vector<Box>& boxes,
                              const Mask& allowed) const {
  Mask valid = allowed;
  std::vector<double> gates;
  for (size_t r = 0; r < tracks.size(); ++r) {
    for (size_t c = 0; c < boxes.size(); ++c) valid[r][c] = valid[r][c] && same_size(tracks[r]->box, boxes[c]);
    gates.push_back(capped(*tracks[r], gate(*tracks[r])));
  }
  return match_by_distance(predicted_boxes(tracks), boxes, valid, gates);
}

Pairs Tracker::match_unmatched(const std::function<bool(const State&)>& wanted, const Matcher& matcher,
                               const std::vector<Detection>& detections, const Mask& allowed,
                               const Pairs& pairs) const {
  std::vector<bool> track_taken(tracks_.size()), det_taken(detections.size());
  for (auto [ti, di] : pairs) {
    track_taken[ti] = true;
    det_taken[di] = true;
  }
  std::vector<int> tis, dis;
  for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti)
    if (wanted(tracks_[ti]) && !track_taken[ti]) tis.push_back(ti);
  for (int di = 0; di < static_cast<int>(detections.size()); ++di)
    if (!det_taken[di]) dis.push_back(di);

  std::vector<const State*> sub_tracks;
  for (int ti : tis) sub_tracks.push_back(&tracks_[ti]);
  std::vector<Box> sub_boxes;
  for (int di : dis) sub_boxes.push_back(detections[di].box);
  Mask sub_allowed(tis.size(), std::vector<bool>(dis.size()));
  for (size_t a = 0; a < tis.size(); ++a)
    for (size_t b = 0; b < dis.size(); ++b) sub_allowed[a][b] = allowed[tis[a]][dis[b]];

  Pairs out;
  for (auto [a, b] : matcher(sub_tracks, sub_boxes, sub_allowed)) out.emplace_back(tis[a], dis[b]);
  return out;
}

}  // namespace tether
