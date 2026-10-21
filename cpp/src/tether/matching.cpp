#include "tether/matching.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tether {

namespace {

// Cost of a pair that can't match. It exceeds any total of valid costs (each is
// at most 1 for IoU, the distance gate for distance), so the solver first
// maximizes the number of valid pairs.
constexpr double INVALID_COST = 1e6;

double area(const Box& b) { return std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]); }

// Solves with invalid cells priced at INVALID_COST, then drops the invalid pairs it chose.
Pairs assign(CostMatrix cost, const Mask& valid) {
  for (size_t r = 0; r < cost.size(); ++r)
    for (size_t c = 0; c < cost[r].size(); ++c)
      if (!valid[r][c]) cost[r][c] = INVALID_COST;
  Pairs pairs;
  for (auto [r, c] : linear_sum_assignment(cost))
    if (valid[r][c]) pairs.emplace_back(r, c);
  return pairs;
}

bool permitted(const Mask& allowed, size_t r, size_t c) { return allowed.empty() || allowed[r][c]; }

}  // namespace

double iou(const Box& a, const Box& b) {
  double inter_w = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  double inter_h = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  double inter = inter_w * inter_h;
  double uni = area(a) + area(b) - inter;
  return uni > 0 ? inter / uni : 0.0;
}

double center_distance(const Box& track, const Box& det) {
  double dx = (det[0] + det[2] - track[0] - track[2]) / 2;
  double dy = (det[1] + det[3] - track[1] - track[3]) / 2;
  double size = std::max(track[2] - track[0], track[3] - track[1]);
  return size > 0 ? std::hypot(dx, dy) / size : std::numeric_limits<double>::infinity();
}

Pairs match(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed, double min_iou) {
  if (tracks.empty() || dets.empty()) return {};
  CostMatrix cost(tracks.size(), std::vector<double>(dets.size()));
  Mask valid(tracks.size(), std::vector<bool>(dets.size()));
  for (size_t r = 0; r < tracks.size(); ++r)
    for (size_t c = 0; c < dets.size(); ++c) {
      double overlap = iou(tracks[r], dets[c]);
      cost[r][c] = 1.0 - overlap;
      valid[r][c] = overlap >= min_iou && permitted(allowed, r, c);
    }
  return assign(std::move(cost), valid);
}

Pairs match_by_distance(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed,
                        const std::vector<double>& max_distance) {
  if (tracks.empty() || dets.empty()) return {};
  CostMatrix cost(tracks.size(), std::vector<double>(dets.size()));
  Mask valid(tracks.size(), std::vector<bool>(dets.size()));
  for (size_t r = 0; r < tracks.size(); ++r)
    for (size_t c = 0; c < dets.size(); ++c) {
      cost[r][c] = center_distance(tracks[r], dets[c]);
      valid[r][c] = cost[r][c] <= max_distance[r] && permitted(allowed, r, c);
    }
  return assign(std::move(cost), valid);
}

Pairs match_by_distance(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed,
                        double max_distance) {
  return match_by_distance(tracks, dets, allowed, std::vector<double>(tracks.size(), max_distance));
}

}  // namespace tether
