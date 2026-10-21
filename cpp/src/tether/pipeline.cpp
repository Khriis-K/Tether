#include "tether/pipeline.hpp"

#include <chrono>
#include <exception>
#include <mutex>
#include <thread>

#include "tether/bounded_queue.hpp"

#ifdef TETHER_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace tether {

namespace {

// Marks a step as a named range on a profiler's timeline (NVTX, which Nsight
// Systems shows per thread) while it's in scope. NVTX does nothing unless a
// profiler is attached; builds without the CUDA toolkit leave it out.
class StepRange {
 public:
#ifdef TETHER_HAS_NVTX
  explicit StepRange(const char* step) { nvtxRangePushA(step); }
  ~StepRange() { nvtxRangePop(); }
#else
  explicit StepRange(const char*) {}
#endif
  StepRange(const StepRange&) = delete;
  StepRange& operator=(const StepRange&) = delete;
};

// Runs `body` as the step named `step` and returns when it started and ended.
template <class Body>
Span timed(const Clock& clock, const char* step, Body&& body) {
  StepRange range(step);
  double start = clock();
  body();
  return Span{start, clock()};
}

// One frame on its way through the parallel stages, gathering each step's result.
struct InFlight {
  cv::Mat frame;
  StepTimes times{};
  PreparedFrame prepared;
  std::vector<Detection> detections;
  std::vector<Track> tracks;
};

using Queue = BoundedQueue<InFlight>;

}  // namespace

double steady_clock_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void run_sequential(VideoSource& source, FramePrep& prep, Detector& detector, Tracker& tracker, const Show& show,
                    const Clock& clock, const std::function<void(const StepTimes&)>& on_frame) {
  auto step = [&](const char* name, auto&& body) { return timed(clock, name, body); };
  cv::Mat frame;
  while (true) {
    StepTimes times{};
    bool got = false;
    times.capture = step("capture", [&] { got = source.read(frame); });
    if (!got) return;
    times.dropped = source.dropped();
    PreparedFrame prepared;
    std::vector<Detection> detections;
    std::vector<Track> tracks;
    bool keep_going = true;
    times.prep = step("prep", [&] { prepared = prep.prepare(frame); });
    times.detect = step("detect", [&] { detections = detector.detect(prepared); });
    times.track = step("track", [&] { tracks = tracker.update(detections); });
    times.draw = step("draw", [&] { keep_going = show(frame, tracks); });
    on_frame(times);
    if (!keep_going) return;
  }
}

void run_parallel(VideoSource& source, FramePrep& prep, Detector& detector, Tracker& tracker, const Show& show,
                  const Clock& clock, const std::function<void(const StepTimes&)>& on_frame) {
  Queue to_prep(PARALLEL_QUEUE_CAPACITY), to_detect(PARALLEL_QUEUE_CAPACITY), to_track(PARALLEL_QUEUE_CAPACITY),
      to_draw(PARALLEL_QUEUE_CAPACITY);
  std::mutex error_mutex;
  std::exception_ptr error;  // the first step to fail
  auto fail = [&] {
    std::lock_guard lock(error_mutex);
    if (!error) error = std::current_exception();
  };

  // A stage that stops, at the end of the video or on an error, closes its
  // output: the stages after it finish the frames already queued, then stop
  // too, and draw ends. The stages before it are stopped by stages.stop().
  auto stage = [&](Queue& in, Queue& out, auto work) {
    return std::thread([&in, &out, work, &fail] {
      try {
        while (auto item = in.pop()) {
          work(*item);
          if (!out.push(std::move(*item))) break;
        }
      } catch (...) {
        fail();
      }
      out.close();
    });
  };

  // stop() closes every queue, so each stage stops at its next push or pop, and
  // waits for all of them. Also run on the way out when draw or on_frame throws.
  struct Stages {
    std::vector<Queue*> queues;
    std::vector<std::thread> threads;
    void stop() {
      for (auto* q : queues) q->close();
      for (auto& t : threads) t.join();
      threads.clear();
    }
    ~Stages() { stop(); }
  } stages{{&to_prep, &to_detect, &to_track, &to_draw}, {}};

  stages.threads.emplace_back([&] {
    try {
      while (true) {
        InFlight item;
        bool got = false;
        item.times.capture = timed(clock, "capture", [&] { got = source.read(item.frame); });
        if (!got) break;
        item.times.dropped = source.dropped();
        if (!to_prep.push(std::move(item))) break;
      }
    } catch (...) {
      fail();
    }
    to_prep.close();
  });
  stages.threads.push_back(stage(to_prep, to_detect, [&](InFlight& f) {
    f.times.prep = timed(clock, "prep", [&] { f.prepared = prep.prepare(f.frame); });
  }));
  stages.threads.push_back(stage(to_detect, to_track, [&](InFlight& f) {
    f.times.detect = timed(clock, "detect", [&] { f.detections = detector.detect(f.prepared); });
  }));
  stages.threads.push_back(stage(to_track, to_draw, [&](InFlight& f) {
    f.times.track = timed(clock, "track", [&] { f.tracks = tracker.update(f.detections); });
  }));

  while (auto f = to_draw.pop()) {
    bool keep_going = true;
    f->times.draw = timed(clock, "draw", [&] { keep_going = show(f->frame, f->tracks); });
    on_frame(f->times);
    if (!keep_going) break;
  }
  stages.stop();
  if (error) std::rethrow_exception(error);
}

void dump_detections(VideoSource& source, FramePrep& prep, Detector& detector,
                     const std::function<void(const std::vector<Detection>&)>& on_frame,
                     const std::function<bool()>& keep_going) {
  cv::Mat frame;
  while (keep_going() && source.read(frame)) on_frame(detector.detect(prep.prepare(frame)));
}

}  // namespace tether
