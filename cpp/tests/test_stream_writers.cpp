// Detection files and timing logs written a frame at a time, as a pipeline run
// produces them (docs/FORMATS.md).

#include <gtest/gtest.h>

#include <stdexcept>

#include "test_util.hpp"
#include "tether/formats.hpp"

using namespace tether;
using tether::test::det;
using tether::test::read_bytes;
using tether::test::temp_path;

namespace {

StepTimes sequential_frame() {
  return {{0.0, 2.0}, {2.0, 3.5}, {3.5, 21.0}, {21.0, 21.25}, {21.25, 23.0}, std::nullopt};
}

}  // namespace

TEST(TimingLogWriter, WritesOneLinePerFrameWithEveryStepInTheSharedFormat) {
  auto path = temp_path("timing.jsonl");
  {
    TimingLogWriter log(path);
    log.write(sequential_frame());
    log.write(sequential_frame());
  }

  std::string line = "\"steps\": {\"capture\": [0.0, 2.0], \"prep\": [2.0, 3.5], \"detect\": [3.5, 21.0], "
                     "\"track\": [21.0, 21.25], \"draw\": [21.25, 23.0]}}\n";
  EXPECT_EQ(read_bytes(path), "{\"frame\": 0, " + line + "{\"frame\": 1, " + line);
}

TEST(TimingLogWriter, AWebcamFrameAlsoRecordsTheFramesDroppedBeforeIt) {
  auto path = temp_path("timing.jsonl");
  auto times = sequential_frame();
  times.dropped = 3;
  {
    TimingLogWriter log(path);
    log.write(times);
  }

  EXPECT_EQ(read_bytes(path),
            "{\"frame\": 0, \"steps\": {\"capture\": [0.0, 2.0], \"prep\": [2.0, 3.5], \"detect\": [3.5, 21.0], "
            "\"track\": [21.0, 21.25], \"draw\": [21.25, 23.0]}, \"dropped\": 3}\n");
}

TEST(TimingLogWriter, EachLineIsOnDiskAsSoonAsItIsWritten) {
  // A run stopped with Ctrl-C (or killed) must still leave every finished frame.
  auto path = temp_path("timing.jsonl");
  TimingLogWriter log(path);

  log.write(sequential_frame());

  EXPECT_EQ(read_bytes(path).substr(0, 13), "{\"frame\": 0, ");
}

TEST(DetectionWriter, WritesTheSameBytesAsSavingTheWholeFile) {
  DetectionFrames frames = {{det({10.0, 20.0, 30.0, 40.0}, 0.91, 0), det({1.5, 2.5, 3.5, 4.5}, 0.12, 2)}, {}};
  auto whole = temp_path("whole.jsonl");
  auto streamed = temp_path("streamed.jsonl");
  save_detections(whole, frames);

  {
    DetectionWriter writer(streamed);
    for (const auto& dets : frames) writer.write(dets);
  }

  EXPECT_EQ(read_bytes(streamed), read_bytes(whole));
}

TEST(DetectionWriter, NothingWrittenIsAnEmptyFile) {
  auto path = temp_path("dets.jsonl");

  { DetectionWriter writer(path); }

  EXPECT_EQ(read_bytes(path), "");
}

TEST(DetectionWriter, EachLineIsOnDiskAsSoonAsItIsWritten) {
  auto path = temp_path("dets.jsonl");
  DetectionWriter writer(path);

  writer.write({});

  EXPECT_EQ(read_bytes(path), "{\"frame\": 0, \"detections\": []}\n");
}

TEST(StreamWriters, AnUnwritablePathIsAnError) {
  auto path = temp_path("no-such-dir") / "out.jsonl";

  EXPECT_THROW(DetectionWriter{path}, std::runtime_error);
  EXPECT_THROW(TimingLogWriter{path}, std::runtime_error);
}
