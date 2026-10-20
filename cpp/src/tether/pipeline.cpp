#include "tether/pipeline.hpp"

#include <chrono>

namespace tether {

double steady_clock_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void run_sequential(VideoSource& source, FramePrep& prep, Detector& detector, Tracker& tracker, const Show& show,
                    const Clock& clock, const std::function<void(const StepTimes&)>& on_frame) {
  // Runs `body` and returns when it started and ended.
  auto step = [&](auto&& body) {
    double start = clock();
    body();
    return Span{start, clock()};
  };
  cv::Mat frame;
  while (true) {
    StepTimes times{};
    bool got = false;
    times.capture = step([&] { got = source.read(frame); });
    if (!got) return;
    times.dropped = source.dropped();
    PreparedFrame prepared;
    std::vector<Detection> detections;
    std::vector<Track> tracks;
    bool keep_going = true;
    times.prep = step([&] { prepared = prep.prepare(frame); });
    times.detect = step([&] { detections = detector.detect(prepared); });
    times.track = step([&] { tracks = tracker.update(detections); });
    times.draw = step([&] { keep_going = show(frame, tracks); });
    on_frame(times);
    if (!keep_going) return;
  }
}

void dump_detections(VideoSource& source, FramePrep& prep, Detector& detector,
                     const std::function<void(const std::vector<Detection>&)>& on_frame,
                     const std::function<bool()>& keep_going) {
  cv::Mat frame;
  while (keep_going() && source.read(frame)) on_frame(detector.detect(prep.prepare(frame)));
}

}  // namespace tether
