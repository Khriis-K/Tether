// The sequential pipeline loop and the detection dump, with fake steps and a
// fake clock. Mirrors python/tests/test_demo.py.

#include <gtest/gtest.h>

#include <map>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline.hpp"

using namespace tether;
using tether::test::det;

namespace {

// Milliseconds that pass only when a fake step says so.
struct FakeClock {
  double now = 0.0;
  void spend(double ms) { now += ms; }
  Clock clock() {
    return [this] { return now; };
  }
};

// Frames tagged with their number in pixel (0, 0); each read takes 2 ms.
class FakeSource : public VideoSource {
 public:
  FakeSource(FakeClock& clock, int n, std::optional<int> dropped = std::nullopt)
      : clock_(clock), n_(n), dropped_(dropped) {}
  bool read(cv::Mat& frame) override {
    clock_.spend(2);
    if (next_ == n_) return false;
    frame = cv::Mat(1, 1, CV_8UC1, cv::Scalar(next_++));
    return true;
  }
  std::optional<int> dropped() const override { return dropped_; }

 private:
  FakeClock& clock_;
  int n_, next_ = 0;
  std::optional<int> dropped_;
};

// Remembers which frame it prepared in frame_height; takes 1 ms.
class FakePrep : public FramePrep {
 public:
  explicit FakePrep(FakeClock& clock) : clock_(clock) {}
  PreparedFrame prepare(const cv::Mat& frame) override {
    clock_.spend(1);
    return {{}, 0, 0, 1.0, 0, 0, frame.at<uchar>(0, 0), 0};
  }

 private:
  FakeClock& clock_;
};

// Returns each frame's scripted detections; takes 10 ms.
class FakeDetector : public Detector {
 public:
  FakeDetector(FakeClock& clock, std::map<int, std::vector<Detection>> per_frame)
      : clock_(clock), per_frame_(std::move(per_frame)) {}
  std::pair<int, int> input_size() const override { return {1, 1}; }
  std::vector<Detection> detect(const PreparedFrame& prepared) override {
    clock_.spend(10);
    seen.push_back(prepared.frame_height);
    return per_frame_[prepared.frame_height];
  }
  std::vector<int> seen;

 private:
  FakeClock& clock_;
  std::map<int, std::vector<Detection>> per_frame_;
};

int tag(const cv::Mat& frame) { return frame.at<uchar>(0, 0); }

}  // namespace

TEST(RunSequential, LogsEveryStepOfEveryFrameInOrder) {
  FakeClock clock;
  FakeSource source(clock, 2);
  FakePrep prep(clock);
  auto d = det({0.0, 0.0, 10.0, 10.0});
  FakeDetector detector(clock, {{0, {d}}, {1, {d}}});
  Tracker tracker;
  std::vector<std::pair<int, std::vector<Box>>> shown;
  std::vector<StepTimes> log;

  run_sequential(
      source, prep, detector, tracker,
      [&](const cv::Mat& frame, const std::vector<Track>& tracks) {
        std::vector<Box> boxes;
        for (const auto& t : tracks) boxes.push_back(t.box);
        shown.emplace_back(tag(frame), boxes);
        clock.spend(3);
        return true;
      },
      clock.clock(), [&](const StepTimes& t) { log.push_back(t); });

  EXPECT_EQ(detector.seen, (std::vector<int>{0, 1}));
  ASSERT_EQ(shown.size(), 2u);
  EXPECT_EQ(shown[1], (std::pair<int, std::vector<Box>>{1, {d.box}}));
  ASSERT_EQ(log.size(), 2u);
  EXPECT_EQ(log[1].capture, (Span{16.0, 18.0}));
  EXPECT_EQ(log[1].prep, (Span{18.0, 19.0}));
  EXPECT_EQ(log[1].detect, (Span{19.0, 29.0}));
  EXPECT_EQ(log[1].track, (Span{29.0, 29.0}));
  EXPECT_EQ(log[1].draw, (Span{29.0, 32.0}));
}

TEST(RunSequential, StopsAfterTheFrameWhereShowSaysStop) {
  FakeClock clock;
  FakeSource source(clock, 3);
  FakePrep prep(clock);
  FakeDetector detector(clock, {});
  Tracker tracker;
  std::vector<int> shown;
  int frames = 0;

  run_sequential(
      source, prep, detector, tracker,
      [&](const cv::Mat& frame, const std::vector<Track>&) {
        shown.push_back(tag(frame));
        return tag(frame) != 1;
      },
      clock.clock(), [&](const StepTimes&) { ++frames; });

  EXPECT_EQ(shown, (std::vector<int>{0, 1}));
  EXPECT_EQ(frames, 2);
}

TEST(RunSequential, EachFrameCarriesTheSourcesDroppedCount) {
  FakeClock clock;
  FakePrep prep(clock);
  FakeDetector detector(clock, {});
  auto show = [](const cv::Mat&, const std::vector<Track>&) { return true; };

  for (std::optional<int> dropped : {std::optional<int>{}, std::optional<int>{2}}) {
    FakeSource source(clock, 1, dropped);
    Tracker tracker;
    std::vector<StepTimes> log;
    run_sequential(source, prep, detector, tracker, show, clock.clock(),
                   [&](const StepTimes& t) { log.push_back(t); });

    ASSERT_EQ(log.size(), 1u);
    EXPECT_EQ(log[0].dropped, dropped);
  }
}

TEST(DumpDetections, GivesEveryFramesDetectionsUnfiltered) {
  FakeClock clock;
  FakeSource source(clock, 3);
  FakePrep prep(clock);
  std::map<int, std::vector<Detection>> per_frame = {
      {0, {det({0.0, 0.0, 10.0, 10.0}, 0.95, 0), det({20.0, 20.0, 30.0, 30.0}, 0.11, 2)}},
      {1, {}},
      {2, {det({5.0, 5.0, 15.0, 15.0}, 0.12, 0)}},
  };
  FakeDetector detector(clock, per_frame);
  DetectionFrames frames;

  dump_detections(source, prep, detector, [&](const std::vector<Detection>& d) { frames.push_back(d); },
                  [] { return true; });

  EXPECT_EQ(frames, (DetectionFrames{per_frame[0], per_frame[1], per_frame[2]}));
}

TEST(DumpDetections, StopsWhenAskedBeforeTheNextFrame) {
  FakeClock clock;
  FakeSource source(clock, 5);
  FakePrep prep(clock);
  FakeDetector detector(clock, {});
  int asked = 0, frames = 0;

  dump_detections(source, prep, detector, [&](const std::vector<Detection>&) { ++frames; },
                  [&] { return ++asked <= 2; });

  EXPECT_EQ(frames, 2);
}
