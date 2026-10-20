// Helpers shared by the tests.
#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace tether::test {

// A path in a fresh directory for the running test, like pytest's tmp_path.
inline std::filesystem::path temp_path(const std::string& name) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  auto dir = std::filesystem::temp_directory_path() / "tether_tests" /
             (std::string(info->test_suite_name()) + "." + info->name());
  static std::string cleaned;  // wipe each test's directory once, on first use
  if (cleaned != dir.string()) {
    std::filesystem::remove_all(dir);
    cleaned = dir.string();
  }
  std::filesystem::create_directories(dir);
  return dir / name;
}

inline std::string read_bytes(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace tether::test
