// The parallel pipeline: each step on its own thread, bounded queues between
// them. Fakes here are safe to call from any thread and use the real clock.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <opencv2/core.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "test_util.hpp"
#include "tether/pipeline.hpp"

using namespace tether;
using namespace std::chrono_literals;
using tether::test::det;

namespace {

void sleep_ms(double ms) { std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms)); }

// Small random waits, so the threads interleave differently on every run.
class Jitter {
 public:
  explicit Jitter(unsigned seed, double max_ms) : rng_(seed), dist_(0.0, max_ms) {}
  void wait() {
    double ms;
    {
      std::lock_guard lock(mutex_);
      ms = dist_(rng_);
    }
    // Spins rather than sleeps: Windows rounds short sleeps up to its timer tick.
    double until = steady_clock_ms() + ms;
    while (steady_clock_ms() < until) std::this_thread::yield();
  }

 private:
  std::mutex mutex_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> dist_;
};

// Frames tagged with their number in pixel (0, 0), as int so long runs fit;
// n < 0 never ends. Each read waits `ms`.
class ThreadSource : public VideoSource {
 public:
  explicit ThreadSource(int n, double ms = 0, std::optional<int> dropped = std::nullopt)
      : n_(n), ms_(ms), dropped_(dropped) {}
  bool read(cv::Mat& frame) override {
    thread = std::this_thread::get_id();
    sleep_ms(ms_);
    if (next_ == n_) return false;
    frame = cv::Mat(1, 1, CV_32SC1, cv::Scalar(next_++));
    ++reads;
    return true;
  }
  std::optional<int> dropped() const override { return dropped_; }
  std::atomic<int> reads = 0;
  std::thread::id thread;

 private:
  int n_, next_ = 0;
  double ms_;
  std::optional<int> dropped_;
};

int tag(const cv::Mat& frame) { return frame.at<int>(0, 0); }

// Carries the frame number in frame_height.
class ThreadPrep : public FramePrep {
 public:
  explicit ThreadPrep(Jitter* jitter = nullptr) : jitter_(jitter) {}
  PreparedFrame prepare(const cv::Mat& frame) override {
    thread = std::this_thread::get_id();
    if (jitter_) jitter_->wait();
    return {{}, 0, 0, 1.0, 0, 0, tag(frame), 0};
  }
  std::thread::id thread;

 private:
  Jitter* jitter_;
};

// Three objects moving across the frame, one of them missing every 7th frame,
// so the tracker's state (and its IDs) depend on frames arriving in order.
std::vector<Detection> scene(int frame) {
  std::vector<Detection> out;
  double x = 5.0 * frame;
  out.push_back(det({x, 10, x + 40, 90}, 0.9));
  if (frame % 7 != 3) out.push_back(det({300 - 3.0 * frame, 200, 340 - 3.0 * frame, 280}, 0.8));
  out.push_back(det({100, 100 + 2.0 * frame, 150, 160 + 2.0 * frame}, frame % 5 == 0 ? 0.3 : 0.7, 2));
  return out;
}

class ThreadDetector : public Detector {
 public:
  explicit ThreadDetector(double ms = 0, Jitter* jitter = nullptr, int throw_at = -1)
      : ms_(ms), jitter_(jitter), throw_at_(throw_at) {}
  std::pair<int, int> input_size() const override { return {1, 1}; }
  std::vector<Detection> detect(const PreparedFrame& prepared) override {
    thread = std::this_thread::get_id();
    if (prepared.frame_height == throw_at_) throw std::runtime_error("detector failed");
    double start = steady_clock_ms();
    sleep_ms(ms_);
    if (jitter_) jitter_->wait();
    worked.push_back({start, steady_clock_ms()});
    return scene(prepared.frame_height);
  }
  std::thread::id thread;
  std::vector<Span> worked;  // when each call did its work, as the detector itself saw it

 private:
  double ms_;
  Jitter* jitter_;
  int throw_at_;
};

struct Shown {
  int frame;
  std::vector<Track> tracks;
};

}  // namespace

TEST(RunParallel, GivesTheSameTracksInTheSameFrameOrderAsSequential) {
  constexpr int N = 120;
  std::vector<Shown> sequential, parallel;
  {
    ThreadSource source(N);
    ThreadPrep prep;
    ThreadDetector detector;
    Tracker tracker;
    run_sequential(source, prep, detector, tracker,
                   [&](const cv::Mat& f, const std::vector<Track>& t) {
                     sequential.push_back({tag(f), t});
                     return true;
                   },
                   steady_clock_ms, [](const StepTimes&) {});
  }
  for (unsigned seed : {1u, 2u, 3u}) {  // a different interleaving each time
    parallel.clear();
    Jitter jitter(seed, 0.5);
    ThreadSource source(N);
    ThreadPrep prep(&jitter);
    ThreadDetector detector(0, &jitter);
    Tracker tracker;
    run_parallel(source, prep, detector, tracker,
                 [&](const cv::Mat& f, const std::vector<Track>& t) {
                   jitter.wait();
                   parallel.push_back({tag(f), t});
                   return true;
                 },
                 steady_clock_ms, [](const StepTimes&) {});

    ASSERT_EQ(parallel.size(), sequential.size()) << "seed " << seed;
    for (int i = 0; i < N; ++i) {
      ASSERT_EQ(parallel[i].frame, i) << "seed " << seed;
      ASSERT_EQ(parallel[i].tracks, sequential[i].tracks) << "seed " << seed << ", frame " << i;
    }
  }
  // The scene makes real tracks, so the comparison isn't of empty lists.
  EXPECT_FALSE(sequential.back().tracks.empty());
}

TEST(RunParallel, RunsEachStepOnItsOwnThreadAndDrawsOnTheCallers) {
  ThreadSource source(5);
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;  // not virtual, so its thread isn't seen here
  std::set<std::thread::id> draw_threads;
  run_parallel(source, prep, detector, tracker,
               [&](const cv::Mat&, const std::vector<Track>&) {
                 draw_threads.insert(std::this_thread::get_id());
                 return true;
               },
               steady_clock_ms, [](const StepTimes&) {});

  auto caller = std::this_thread::get_id();
  std::set<std::thread::id> steps = {source.thread, prep.thread, detector.thread, caller};
  EXPECT_EQ(steps.size(), 4u);  // all different
  EXPECT_EQ(draw_threads, (std::set<std::thread::id>{caller}));  // a window must be drawn where it was made
}

TEST(RunParallel, CallsOnFrameOnTheCallersThreadForEveryFrameInOrder) {
  ThreadSource source(30);
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;
  std::vector<int> shown;
  std::set<std::thread::id> threads;
  int frames = 0;
  run_parallel(source, prep, detector, tracker,
               [&](const cv::Mat& f, const std::vector<Track>&) {
                 shown.push_back(tag(f));
                 return true;
               },
               steady_clock_ms, [&](const StepTimes&) {
                 threads.insert(std::this_thread::get_id());
                 ++frames;
               });

  EXPECT_EQ(frames, 30);
  EXPECT_EQ(shown.size(), 30u);
  EXPECT_EQ(threads, (std::set<std::thread::id>{std::this_thread::get_id()}));
}

TEST(RunParallel, TimesEachStepWhereItRanAndInStepOrderWithinAFrame) {
  ThreadSource source(8);  // instant, so frames do queue up in front of the slower detect
  ThreadPrep prep;
  ThreadDetector detector(10);
  Tracker tracker;
  std::vector<StepTimes> log;
  run_parallel(source, prep, detector, tracker,
               [&](const cv::Mat&, const std::vector<Track>&) {
                 sleep_ms(4);
                 return true;
               },
               steady_clock_ms, [&](const StepTimes& t) { log.push_back(t); });

  ASSERT_EQ(log.size(), 8u);
  for (const auto& t : log) {
    const Span* steps[] = {&t.capture, &t.prep, &t.detect, &t.track, &t.draw};
    for (int i = 0; i < 5; ++i) {
      EXPECT_LE((*steps[i])[0], (*steps[i])[1]);
      if (i > 0) EXPECT_LE((*steps[i - 1])[1], (*steps[i])[0]) << "step " << i << " started before the one before it ended";
    }
    EXPECT_GE(t.draw[1] - t.draw[0], 3.5);
  }
  // The detect span covers the detector's own work, not the wait in the queue
  // before it: detect is the slowest step here, so each frame does wait there.
  ASSERT_EQ(detector.worked.size(), log.size());
  for (size_t i = 0; i < log.size(); ++i) {
    EXPECT_LE(log[i].detect[0], detector.worked[i][0]);
    EXPECT_LT(detector.worked[i][0] - log[i].detect[0], 1.0) << "frame " << i;
    EXPECT_LE(detector.worked[i][1], log[i].detect[1]);
    EXPECT_LT(log[i].detect[1] - detector.worked[i][1], 1.0) << "frame " << i;
  }
  for (size_t i = 1; i < log.size(); ++i) EXPECT_LT(log[i - 1].draw[1], log[i].draw[1]);
}

TEST(RunParallel, OverlapsTheStepsSoFramesFinishFasterThanSequential) {
  // Three 20 ms steps: one after another that's 60 ms a frame; overlapped,
  // the slowest step's 20 ms.
  constexpr int N = 20;
  auto run = [&](bool parallel) {
    ThreadSource source(N, 20);
    ThreadPrep prep;
    ThreadDetector detector(20);
    Tracker tracker;
    auto show = [](const cv::Mat&, const std::vector<Track>&) {
      sleep_ms(20);
      return true;
    };
    double start = steady_clock_ms();
    if (parallel)
      run_parallel(source, prep, detector, tracker, show, steady_clock_ms, [](const StepTimes&) {});
    else
      run_sequential(source, prep, detector, tracker, show, steady_clock_ms, [](const StepTimes&) {});
    return steady_clock_ms() - start;
  };
  double sequential = run(false), parallel = run(true);
  EXPECT_GE(sequential, N * 60.0 * 0.95);
  EXPECT_LT(parallel, sequential * 0.6) << parallel << " ms vs " << sequential << " ms";
}

TEST(RunParallel, EachFrameCarriesTheSourcesDroppedCount) {
  for (std::optional<int> dropped : {std::optional<int>{}, std::optional<int>{2}}) {
    ThreadSource source(3, 0, dropped);
    ThreadPrep prep;
    ThreadDetector detector;
    Tracker tracker;
    std::vector<StepTimes> log;
    run_parallel(source, prep, detector, tracker, [](const cv::Mat&, const std::vector<Track>&) { return true; },
                 steady_clock_ms, [&](const StepTimes& t) { log.push_back(t); });
    ASSERT_EQ(log.size(), 3u);
    for (const auto& t : log) EXPECT_EQ(t.dropped, dropped);
  }
}

TEST(RunParallel, AnEmptySourceEndsWithoutAFrame) {
  ThreadSource source(0);
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;
  int frames = 0;
  run_parallel(source, prep, detector, tracker, [&](const cv::Mat&, const std::vector<Track>&) { return true; },
               steady_clock_ms, [&](const StepTimes&) { ++frames; });
  EXPECT_EQ(frames, 0);
}

TEST(RunParallel, StopsAfterTheFrameWhereShowSaysStopEvenOnAnEndlessSource) {
  ThreadSource source(-1);  // a webcam: never ends on its own
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;
  std::vector<int> shown;
  int frames = 0;
  run_parallel(source, prep, detector, tracker,
               [&](const cv::Mat& f, const std::vector<Track>&) {
                 shown.push_back(tag(f));
                 return tag(f) != 4;
               },
               steady_clock_ms, [&](const StepTimes&) { ++frames; });

  EXPECT_EQ(shown, (std::vector<int>{0, 1, 2, 3, 4}));
  EXPECT_EQ(frames, 5);
}

TEST(RunParallel, ReadsOnlyABoundedNumberOfFramesAheadOfASlowDraw) {
  ThreadSource source(-1);
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;
  int reads_while_held = 0;
  run_parallel(source, prep, detector, tracker,
               [&](const cv::Mat&, const std::vector<Track>&) {
                 sleep_ms(200);  // a paused window: every stage before has time to fill up
                 reads_while_held = source.reads;
                 return false;
               },
               steady_clock_ms, [](const StepTimes&) {});

  // The frame being drawn, one held by each of the four other steps, and a full
  // queue in front of each step after capture.
  int most = 1 + 4 + 4 * static_cast<int>(PARALLEL_QUEUE_CAPACITY);
  EXPECT_LE(reads_while_held, most);
  EXPECT_GE(reads_while_held, 2);  // it did read ahead
}

TEST(RunParallel, AStepThatThrowsEndsTheRunAndTheErrorReachesTheCaller) {
  ThreadSource source(-1);
  ThreadPrep prep;
  ThreadDetector detector(0, nullptr, 6);
  Tracker tracker;
  std::vector<int> shown;
  EXPECT_THROW(run_parallel(source, prep, detector, tracker,
                            [&](const cv::Mat& f, const std::vector<Track>&) {
                              shown.push_back(tag(f));
                              return true;
                            },
                            steady_clock_ms, [](const StepTimes&) {}),
               std::runtime_error);
  EXPECT_EQ(shown, (std::vector<int>{0, 1, 2, 3, 4, 5}));  // every frame before the failed one
}

TEST(RunParallel, AnErrorInDrawEndsTheRunAndReachesTheCaller) {
  ThreadSource source(-1);
  ThreadPrep prep;
  ThreadDetector detector;
  Tracker tracker;
  EXPECT_THROW(run_parallel(source, prep, detector, tracker,
                            [](const cv::Mat& f, const std::vector<Track>&) -> bool {
                              if (tag(f) == 2) throw std::runtime_error("window closed under us");
                              return true;
                            },
                            steady_clock_ms, [](const StepTimes&) {}),
               std::runtime_error);
}
