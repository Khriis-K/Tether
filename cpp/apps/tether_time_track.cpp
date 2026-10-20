#include <iostream>
#include <string>
#include <vector>

#include "tether/timing.hpp"

int main(int argc, char** argv) {
  return tether::run_time_cli(std::vector<std::string>(argv + 1, argv + argc), std::cout, std::cerr);
}
