// tether-track: run the tracker on a detection file and write a track file.
// Mirrors the Python CLI (python/tether/cli.py), minus its diagnostic --single-target.
#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace tether {

// `args` excludes the program name. Returns the process exit code: 0 on
// success, 2 on bad arguments (as argparse), 1 when a file can't be read or written.
int run_track_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace tether
