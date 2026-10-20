// The HUD: tracks drawn onto the video. Status reads from line style alone, so
// it doesn't depend on telling colours apart:
//   tracked    solid corner brackets and an ID label, in the track's colour
//   predicted  thin dashes where a tracked object should be next frame
//   searching  thick accent-coloured dashes where the tracker is looking, and
//              a label with how many frames the object has been missing
// Tentative tracks aren't drawn: they may be one-frame noise.
#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "tether/formats.hpp"

namespace tether {

inline const cv::Scalar ACCENT{0, 106, 255};      // signal orange (BGR): searching and live indicators
inline const cv::Scalar NEAR_BLACK{20, 18, 16};   // label and badge backgrounds
inline const cv::Scalar OFF_WHITE{235, 238, 240};

// A track's hue, as a fraction of the colour wheel. Each new ID steps by the
// golden ratio, so any number of tracks stays spread around the wheel without
// knowing the count in advance, and neighbouring IDs are far apart. Skips the
// band around the accent, which means "searching".
double track_hue(int id);
// That hue as a bright, saturated BGR colour.
cv::Scalar track_color(int id);

// Draws `tracks` onto `frame`, in place.
void draw_hud(cv::Mat& frame, const std::vector<Track>& tracks);

// Marks the corner of `frame` with PAUSED and a REC indicator, as they apply.
void draw_status(cv::Mat& frame, bool paused, bool recording);

}  // namespace tether
