#include <iostream>
#include <string>
#include <vector>

#include "tether/pipeline_cli.hpp"

int main(int argc, char** argv) {
  return tether::run_pipeline_cli(std::vector<std::string>(argv + 1, argv + argc), std::cout, std::cerr);
}
