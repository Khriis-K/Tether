// tether-time-track: time the tracking step alone, for the Python-vs-C++ speed
// check (python/tether/parity.py). File reading and writing are left out.
#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "tether/formats.hpp"

namespace tether {

// Wall time in milliseconds of a fresh default tracker run over every frame,
// once per repeat.
std::vector<double> time_tracking(const DetectionFrames& frames, int repeats);

// `args` excludes the program name: a detection file and an optional
// --repeats N (default 5). Prints {"frames": N, "repeats_ms": [...]} as one
// JSON line. Exit codes as tether-track: 0 ok, 2 bad arguments, 1 unreadable file.
int run_time_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace tether
