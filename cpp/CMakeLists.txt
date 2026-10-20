cmake_minimum_required(VERSION 3.21)  # 3.21: $<TARGET_RUNTIME_DLLS>
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

# The pipeline's video and image work: OpenCV 5.0.0, the version tether-demo's
# opencv-python runs, built from source with only the modules we use. OpenCV
# doesn't build as a subproject, so the first configure builds and installs it
# into the build folder on its own (several minutes), then finds it like any
# installed OpenCV. Static, so there are no OpenCV DLLs to ship; only its FFmpeg
# plugin, which tether_copy_runtime_dlls puts next to our programs.
FetchContent_Declare(opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/5.0.0.tar.gz
  URL_HASH SHA256=b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095
  SOURCE_SUBDIR not-a-subproject)  # download only; built below
FetchContent_MakeAvailable(opencv)
set(OPENCV_INSTALL ${CMAKE_BINARY_DIR}/_deps/opencv-install)
get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(multi_config)
  set(opencv_configs Release Debug)  # both, so Debug builds of ours link a Debug runtime
else()
  set(opencv_configs Release)
endif()
if(NOT EXISTS ${OPENCV_INSTALL}/done)
  message(STATUS "Building OpenCV (${opencv_configs}); this takes several minutes the first time")
  set(opencv_build ${CMAKE_BINARY_DIR}/_deps/opencv-build)
  set(generator -G ${CMAKE_GENERATOR})
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND generator -A ${CMAKE_GENERATOR_PLATFORM})
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND generator -T ${CMAKE_GENERATOR_TOOLSET})
  endif()
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${opencv_SOURCE_DIR} -B ${opencv_build} ${generator} -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL} -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui
      -DBUILD_SHARED_LIBS=OFF -DBUILD_WITH_STATIC_CRT=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF
      -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF
      -DWITH_CUDA=OFF -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF
    COMMAND_ERROR_IS_FATAL ANY)
  foreach(config ${opencv_configs})
    execute_process(COMMAND ${CMAKE_COMMAND} --build ${opencv_build} --config ${config} --target install --parallel
      COMMAND_ERROR_IS_FATAL ANY)
  endforeach()
  file(TOUCH ${OPENCV_INSTALL}/done)
endif()
set(OpenCV_STATIC ON)
find_package(OpenCV 5.0.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui
  PATHS ${OPENCV_INSTALL} NO_DEFAULT_PATH)
file(GLOB_RECURSE OPENCV_FFMPEG_PLUGIN ${OPENCV_INSTALL}/opencv_videoio_ffmpeg*_64.dll)

# The detector's runtime: ONNX Runtime 1.31.0 (CPU), the version tether-demo's
# onnxruntime runs, as Microsoft's prebuilt release.
if(WIN32)
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip
    URL_HASH SHA256=3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195)
else()
  FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-linux-x64-1.31.0.tgz
    URL_HASH SHA256=cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57)
endif()
FetchContent_MakeAvailable(onnxruntime)  # headers and libraries only, nothing to build
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/include)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll
    IMPORTED_IMPLIB ${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib)
else()
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
endif()

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
  src/tether/timing.cpp
  src/tether/tracker.cpp
)
target_include_directories(tether_core PUBLIC src)
target_link_libraries(tether_core PRIVATE nlohmann_json::nlohmann_json)

# The pipeline: video source, frame prep, detector and the sequential loop.
add_library(tether_pipeline
  src/tether/detector.cpp
  src/tether/frame_prep.cpp
  src/tether/hud.cpp
  src/tether/hud_view.cpp
  src/tether/pipeline.cpp
  src/tether/pipeline_cli.cpp
  src/tether/video_source.cpp
)
target_include_directories(tether_pipeline PUBLIC src)
target_link_libraries(tether_pipeline
  PUBLIC tether_core opencv_core opencv_imgproc opencv_videoio opencv_highgui
  PRIVATE onnxruntime)

# Copies a target's runtime DLLs (ONNX Runtime, OpenCV's FFmpeg plugin) next to
# it, so Windows loads ours and not the older onnxruntime.dll in System32.
function(tether_copy_runtime_dlls target)
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> ${OPENCV_FFMPEG_PLUGIN}
        $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS)
  endif()
endfunction()

add_executable(tether_pipeline_app apps/tether_pipeline.cpp)
target_link_libraries(tether_pipeline_app PRIVATE tether_pipeline)
set_target_properties(tether_pipeline_app PROPERTIES OUTPUT_NAME tether-pipeline)
tether_copy_runtime_dlls(tether_pipeline_app)

add_executable(tether_track apps/tether_track.cpp)
target_link_libraries(tether_track PRIVATE tether_core)
set_target_properties(tether_track PROPERTIES OUTPUT_NAME tether-track)

add_executable(tether_time_track apps/tether_time_track.cpp)
target_link_libraries(tether_time_track PRIVATE tether_core)
set_target_properties(tether_time_track PROPERTIES OUTPUT_NAME tether-time-track)

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
    tests/test_detector.cpp
    tests/test_drop_counter.cpp
    tests/test_formats.cpp
    tests/test_frames_dropped.cpp
    tests/test_frame_prep.cpp
    tests/test_hud.cpp
    tests/test_hud_view.cpp
    tests/test_lifecycle.cpp
    tests/test_low_score.cpp
    tests/test_matching.cpp
    tests/test_merge.cpp
    tests/test_pipeline.cpp
    tests/test_pipeline_cli.cpp
    tests/test_pipeline_cli_hud.cpp
    tests/test_pipeline_cli_outputs.cpp
    tests/test_search_gate.cpp
    tests/test_stream_writers.cpp
    tests/test_timing.cpp
    tests/test_tracker.cpp
    tests/test_video_source.cpp
    tests/test_video_source_fps.cpp
  )
  target_link_libraries(tether_tests PRIVATE tether_core tether_pipeline opencv_imgcodecs
    nlohmann_json::nlohmann_json GTest::gtest_main)
  target_compile_definitions(tether_tests PRIVATE TETHER_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx")
  tether_copy_runtime_dlls(tether_tests)
  include(GoogleTest)
  gtest_discover_tests(tether_tests DISCOVERY_MODE PRE_TEST)

  # The Python-vs-C++ match check on the scripted scenes, against the binaries just built.
  # Needs the Python tracker's dependencies (numpy, scipy). Real files: run tether-parity by hand.
  find_package(Python3 COMPONENTS Interpreter)
  if(NOT Python3_FOUND)
    message(WARNING "No Python found: the Python-vs-C++ check (parity_scripted) is not added")
  else()
    add_test(NAME parity_scripted
      COMMAND Python3::Interpreter -m tether.parity --scripted --repeats 1
        --cpp-dir $<TARGET_FILE_DIR:tether_track> --out ${CMAKE_CURRENT_BINARY_DIR}/parity-scripted.json)
    set_tests_properties(parity_scripted PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python")

    # Both whole pipelines on the same frames, through the stand-in model. Needs
    # the Python pipeline's dependencies too (onnxruntime, opencv-python, pytest).
    add_test(NAME pipeline_compare
      COMMAND Python3::Interpreter -m pytest -q -p no:cacheprovider
        ${CMAKE_CURRENT_SOURCE_DIR}/../python/tests/cpp_pipeline_check.py)
    set_tests_properties(pipeline_compare PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python;TETHER_CPP_DIR=$<TARGET_FILE_DIR:tether_pipeline_app>")
  endif()
endif()
