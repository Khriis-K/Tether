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
      -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL} -DBUILD_LIST=core,imgproc,imgcodecs,videoio
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
find_package(OpenCV 5.0.0 EXACT REQUIRED COMPONENTS core imgproc imgcodecs videoio
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

# The optional TensorRT runtime (--runtime tensorrt). NVIDIA's download needs a
# login, so it isn't fetched: unpack a TensorRT release and point TensorRT_ROOT
# at it. Left empty, everything builds without it. Its CUDA runtime is linked
# statically, so whichever CUDA DLLs come first on the PATH don't matter; pass
# CUDAToolkit_ROOT to pick the toolkit TensorRT was built for.
set(TensorRT_ROOT "" CACHE PATH "TensorRT release folder (include/, lib/, bin/); empty: build without TensorRT")
if(TensorRT_ROOT)
  find_package(CUDAToolkit REQUIRED)
  find_path(TensorRT_INCLUDE_DIR NvInferRuntime.h PATHS ${TensorRT_ROOT}/include NO_DEFAULT_PATH REQUIRED)
  find_library(TensorRT_LIBRARY NAMES nvinfer_11 nvinfer_10 nvinfer PATHS ${TensorRT_ROOT}/lib NO_DEFAULT_PATH
    REQUIRED)
  find_program(TensorRT_TRTEXEC trtexec PATHS ${TensorRT_ROOT}/bin NO_DEFAULT_PATH REQUIRED)
  add_library(tensorrt SHARED IMPORTED)
  set_target_properties(tensorrt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR})
  if(WIN32)
    # The DLL is copied next to our programs with ONNX Runtime's (about 400 MB, once per build folder).
    get_filename_component(trt_lib_name ${TensorRT_LIBRARY} NAME_WE)
    find_file(TensorRT_DLL ${trt_lib_name}.dll PATHS ${TensorRT_ROOT}/bin NO_DEFAULT_PATH REQUIRED)
    set_target_properties(tensorrt PROPERTIES IMPORTED_LOCATION ${TensorRT_DLL} IMPORTED_IMPLIB ${TensorRT_LIBRARY})
  else()
    set_target_properties(tensorrt PROPERTIES IMPORTED_LOCATION ${TensorRT_LIBRARY})
  endif()
  message(STATUS "TensorRT runtime: ${TensorRT_LIBRARY}, CUDA ${CUDAToolkit_VERSION}")
else()
  message(STATUS "TensorRT runtime: off (set TensorRT_ROOT to build it)")
endif()

# The stats panel's window: Dear ImGui drawn through OpenGL 3 in a GLFW window.
# ImGui ships no CMake, so its sources are built here as one library.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9)
FetchContent_Declare(imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9b.tar.gz
  URL_HASH SHA256=8e1bbc76c71d74fef2fb85db7e7ca8eba13d6a86623c54992b60162db554ffdb)
FetchContent_MakeAvailable(glfw imgui)
find_package(OpenGL REQUIRED)
add_library(imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# The panel's fonts, IBM Plex Mono and Sans (SIL Open Font License), built into
# the program as byte arrays so it runs from any folder.
FetchContent_Declare(plex_mono
  URL https://github.com/IBM/plex/releases/download/%40ibm/plex-mono%402.5.0/ibm-plex-mono.zip
  URL_HASH SHA256=6d23f01257663d8cc49a0d64c22ced630b79e0e2a0ac08a0da86e9a38bbc481c
  SOURCE_SUBDIR not-a-subproject)
FetchContent_Declare(plex_sans
  URL https://github.com/IBM/plex/releases/download/%40ibm/plex-sans%401.1.0/ibm-plex-sans.zip
  URL_HASH SHA256=fb365d910566e6d199cc2c15579a7dd9a267128e18431a394ed81f1970c69200
  SOURCE_SUBDIR not-a-subproject)
FetchContent_MakeAvailable(plex_mono plex_sans)
set(PANEL_FONTS_CPP ${CMAKE_BINARY_DIR}/generated/panel_fonts.cpp)
set(PANEL_FONTS_FILES
  "PLEX_SANS_REGULAR|${plex_sans_SOURCE_DIR}/fonts/complete/ttf/IBMPlexSans-Regular.ttf"
  "PLEX_MONO_REGULAR|${plex_mono_SOURCE_DIR}/fonts/complete/ttf/IBMPlexMono-Regular.ttf"
  "PLEX_MONO_SEMIBOLD|${plex_mono_SOURCE_DIR}/fonts/complete/ttf/IBMPlexMono-SemiBold.ttf")
# Generated again only when the fonts (a new download) or this file are newer: it takes a while.
if(${plex_mono_SOURCE_DIR}/LICENSE.txt IS_NEWER_THAN ${PANEL_FONTS_CPP}
   OR ${plex_sans_SOURCE_DIR}/LICENSE.txt IS_NEWER_THAN ${PANEL_FONTS_CPP}
   OR ${CMAKE_CURRENT_LIST_FILE} IS_NEWER_THAN ${PANEL_FONTS_CPP})
  set(fonts_cpp "// Generated by CMakeLists.txt from the IBM Plex releases.\n#include \"tether/panel_fonts.hpp\"\n")
  foreach(font ${PANEL_FONTS_FILES})
    string(REPLACE "|" ";" font ${font})
    list(GET font 0 name)
    list(GET font 1 file)
    file(READ ${file} hex HEX)
    string(REGEX REPLACE "(..)" "0x\\1," bytes ${hex})
    string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n" bytes "${bytes}")  # 16 a line
    string(LENGTH ${hex} hex_length)
    math(EXPR size "${hex_length} / 2")
    string(APPEND fonts_cpp "static const unsigned char ${name}_BYTES[] = {${bytes}};\n"
      "const tether::FontFile tether::${name}{${name}_BYTES, ${size}};\n")
  endforeach()
  file(WRITE ${PANEL_FONTS_CPP} "${fonts_cpp}")
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
  src/tether/live_stats.cpp
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
  src/tether/gui_screen.cpp
  src/tether/hud.cpp
  src/tether/hud_view.cpp
  src/tether/panel.cpp
  src/tether/pipeline.cpp
  src/tether/pipeline_cli.cpp
  src/tether/video_source.cpp
  ${PANEL_FONTS_CPP}
)
target_include_directories(tether_pipeline PUBLIC src)
target_link_libraries(tether_pipeline
  PUBLIC tether_core opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio imgui
  PRIVATE onnxruntime)
if(TensorRT_ROOT)
  target_sources(tether_pipeline PRIVATE src/tether/tensorrt_detector.cpp)
  target_compile_definitions(tether_pipeline PUBLIC TETHER_HAS_TENSORRT)
  target_link_libraries(tether_pipeline PRIVATE tensorrt CUDA::cudart_static)
endif()

# Copies a target's runtime DLLs (ONNX Runtime, TensorRT, OpenCV's FFmpeg plugin) next to
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
    tests/test_bounded_queue.cpp
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
    tests/test_live_stats.cpp
    tests/test_low_score.cpp
    tests/test_matching.cpp
    tests/test_merge.cpp
    tests/test_panel.cpp
    tests/test_parallel_pipeline.cpp
    tests/test_pipeline.cpp
    tests/test_pipeline_cli.cpp
    tests/test_pipeline_cli_hud.cpp
    tests/test_pipeline_cli_mode.cpp
    tests/test_pipeline_cli_outputs.cpp
    tests/test_pipeline_cli_screenshot.cpp
    tests/test_run_settings.cpp
    tests/test_runtime_setting.cpp
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
  if(TensorRT_ROOT)
    # The stand-in models' engines, built by trtexec when the model is newer.
    target_sources(tether_tests PRIVATE tests/test_tensorrt_detector.cpp)
    set(test_engines ${CMAKE_CURRENT_BINARY_DIR}/test-engines)
    add_custom_command(OUTPUT ${test_engines}/fixed_736x1280.engine
      COMMAND ${CMAKE_COMMAND} -E make_directory ${test_engines}
      COMMAND ${TensorRT_TRTEXEC} --onnx=${CMAKE_CURRENT_SOURCE_DIR}/tests/onnx/fixed_736x1280.onnx
        --saveEngine=${test_engines}/fixed_736x1280.engine --noTF32 --skipInference > ${test_engines}/trtexec.log
      DEPENDS tests/onnx/fixed_736x1280.onnx
      COMMENT "Building the stand-in model's TensorRT engine")
    add_custom_target(tether_test_engines DEPENDS ${test_engines}/fixed_736x1280.engine)
    add_dependencies(tether_tests tether_test_engines)
    target_compile_definitions(tether_tests PRIVATE TETHER_TEST_ENGINES="${test_engines}")
  endif()
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
