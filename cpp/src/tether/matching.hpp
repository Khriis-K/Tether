// Pairing detections with tracks: IoU or center-distance cost, Hungarian assignment.
#pragma once

#include <vector>

#include "tether/assignment.hpp"
#include "tether/formats.hpp"

namespace tether {

inline constexpr double MIN_IOU = 0.3;              // pairs that overlap less are never matched by IoU
inline constexpr double MAX_CENTER_DISTANCE = 4.0;  // in track box sizes; farther pairs never match by distance

// allowed[track][detection]: which pairs may match at all. Empty allows every pair.
using Mask = std::vector<std::vector<bool>>;

double iou(const Box& a, const Box& b);

// Distance between box centers, in units of the track box's larger side.
double center_distance(const Box& track, const Box& det);

// Best one-to-one pairing of tracks to detections, as (track, detection) index
// pairs. A pair is valid when its IoU is at least `min_iou` and `allowed`
// permits it. Returns the assignment with the most valid pairs, and among
// those the lowest total (1 - IoU). Sorted by track index.
Pairs match(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed = {},
            double min_iou = MIN_IOU);

// Like `match`, but by center distance, for boxes that may not overlap at all.
// A pair is valid when its center_distance is at most the track's gate:
// `max_distance` holds one per track.
Pairs match_by_distance(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed,
                        const std::vector<double>& max_distance);
// One gate for every track.
Pairs match_by_distance(const std::vector<Box>& tracks, const std::vector<Box>& dets, const Mask& allowed = {},
                        double max_distance = MAX_CENTER_DISTANCE);

}  // namespace tether
