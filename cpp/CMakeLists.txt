cmake_minimum_required(VERSION 3.20)
project(tether LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
# Pinned releases, fetched at configure time.
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(tether_core
  src/tether/assignment.cpp
  src/tether/cli.cpp
  src/tether/formats.cpp
  src/tether/kalman.cpp
  src/tether/matching.cpp
  src/tether/tracker.cpp
)
target_include_directories(tether_core PUBLIC src)
target_link_libraries(tether_core PRIVATE nlohmann_json::nlohmann_json)

add_executable(tether_track apps/tether_track.cpp)
target_link_libraries(tether_track PRIVATE tether_core)
set_target_properties(tether_track PROPERTIES OUTPUT_NAME tether-track)

option(TETHER_BUILD_TESTS "Build the tests" ON)
if(TETHER_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match the MSVC runtime of our targets
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(tether_tests
    tests/test_assignment.cpp
    tests/test_cli.cpp
    tests/test_coast_damping.cpp
    tests/test_formats.cpp
    tests/test_lifecycle.cpp
    tests/test_low_score.cpp
    tests/test_matching.cpp
    tests/test_merge.cpp
    tests/test_search_gate.cpp
    tests/test_tracker.cpp
  )
  target_link_libraries(tether_tests PRIVATE tether_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(tether_tests)
endif()
