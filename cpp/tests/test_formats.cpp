// Detection and track files: the shared JSON Lines formats (docs/FORMATS.md).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "test_util.hpp"
#include "tether/formats.hpp"

using tether::Detection;
using tether::Track;
using tether::test::read_bytes;
using tether::test::temp_path;

TEST(Formats, DetectionsRoundTripIncludingEmptyFrames) {
  std::vector<std::vector<Detection>> frames = {
      {{{10.0, 20.0, 30.0, 40.0}, 0.9, 0}},
      {},
      {{{1.5, 2.5, 3.5, 4.5}, 0.12, 2}, {{5.0, 6.0, 7.0, 8.0}, 0.5, 0}},
  };
  auto path = temp_path("dets.jsonl");

  tether::save_detections(path, frames);

  EXPECT_EQ(tether::load_detections(path), frames);
}

TEST(Formats, TracksRoundTripIncludingEmptyFrames) {
  std::vector<std::vector<Track>> frames = {
      {{1, {10.0, 20.0, 30.0, 40.0}, 0, "tracked", {11.0, 21.0, 31.0, 41.0}, 0,
        {{20.0, 30.0}, {21.0, 31.0}}}},
      {},
  };
  auto path = temp_path("tracks.jsonl");

  tether::save_tracks(path, frames);

  EXPECT_EQ(tether::load_tracks(path), frames);
}

TEST(Formats, FilesUseLfLineEndingsOnEveryPlatform) {
  auto dets_path = temp_path("dets.jsonl");
  auto tracks_path = temp_path("tracks.jsonl");

  tether::save_detections(dets_path, {{}, {}});
  tether::save_tracks(tracks_path, {{}, {}});

  EXPECT_EQ(read_bytes(dets_path),
            "{\"frame\": 0, \"detections\": []}\n{\"frame\": 1, \"detections\": []}\n");
  EXPECT_EQ(read_bytes(tracks_path), "{\"frame\": 0, \"tracks\": []}\n{\"frame\": 1, \"tracks\": []}\n");
}

TEST(Formats, DetectionFileIsOneJsonLinePerFrame) {
  auto path = temp_path("dets.jsonl");

  tether::save_detections(path, {{{{1.0, 2.0, 3.0, 4.0}, 0.5, 7}}, {}});

  EXPECT_EQ(read_bytes(path),
            "{\"frame\": 0, \"detections\": [{\"box\": [1.0, 2.0, 3.0, 4.0], \"score\": 0.5, \"class\": 7}]}\n"
            "{\"frame\": 1, \"detections\": []}\n");
}

TEST(Formats, NumbersArePrintedTheWayPythonPrintsThem) {
  // Expected text is Python's json.dumps of the same values, so both trackers'
  // files read alike: shortest round-trip digits, exponents outside 1e-4..1e16.
  auto path = temp_path("dets.jsonl");

  tether::save_detections(path, {{{{0.1, 1e-05, 123456789012345678.0, 2.5e-7}, 0.30000000000000004, 3},
                                  {{1.0 / 3.0, 100.0, -0.0, 1e16}, 1e15, 0},
                                  {{0.0001, 0.0, 1.0, 2.0}, 1.0, 0}}});

  EXPECT_EQ(read_bytes(path),
            "{\"frame\": 0, \"detections\": ["
            "{\"box\": [0.1, 1e-05, 1.2345678901234568e+17, 2.5e-07], \"score\": 0.30000000000000004, \"class\": 3}, "
            "{\"box\": [0.3333333333333333, 100.0, -0.0, 1e+16], \"score\": 1000000000000000.0, \"class\": 0}, "
            "{\"box\": [0.0001, 0.0, 1.0, 2.0], \"score\": 1.0, \"class\": 0}]}\n");
}

TEST(Formats, TrackLineHasEveryFieldInTheSharedOrder) {
  auto path = temp_path("tracks.jsonl");

  tether::save_tracks(path, {{{3, {1.0, 2.0, 3.0, 4.0}, 2, "searching", {1.5, 2.0, 3.5, 4.0}, 4, {{2.0, 3.0}}}}});

  EXPECT_EQ(read_bytes(path),
            "{\"frame\": 0, \"tracks\": [{\"id\": 3, \"box\": [1.0, 2.0, 3.0, 4.0], \"class\": 2, "
            "\"status\": \"searching\", \"predicted_box\": [1.5, 2.0, 3.5, 4.0], \"frames_since_seen\": 4, "
            "\"path\": [[2.0, 3.0]]}]}\n");
}

TEST(Formats, ReadsIntegerNumbersAndSkipsBlankLines) {
  // Files written by hand or by other tools may print 10 instead of 10.0.
  auto path = temp_path("dets.jsonl");
  {
    std::ofstream f(path, std::ios::binary);
    f << "{\"frame\": 0, \"detections\": [{\"box\": [10, 20, 30, 40], \"score\": 1, \"class\": 0}]}\r\n\n";
  }

  auto frames = tether::load_detections(path);

  ASSERT_EQ(frames.size(), 1u);
  EXPECT_EQ(frames[0], (std::vector<Detection>{{{10.0, 20.0, 30.0, 40.0}, 1.0, 0}}));
}

TEST(Formats, FrameNumbersOutOfOrderAreAnError) {
  auto path = temp_path("dets.jsonl");
  {
    std::ofstream f(path, std::ios::binary);
    f << "{\"frame\": 0, \"detections\": []}\n{\"frame\": 2, \"detections\": []}\n";
  }

  EXPECT_THROW(tether::load_detections(path), std::runtime_error);
}

TEST(Formats, MissingFileIsAnError) {
  EXPECT_THROW(tether::load_detections(temp_path("does-not-exist.jsonl")), std::runtime_error);
}
