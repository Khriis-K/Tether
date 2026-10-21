// Frame prep: turns a raw video frame into the detector's input.
//
// Mirrors python/tether/detector.py (letterbox, prepare) so both pipelines feed
// the model the same numbers.
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace tether {

inline constexpr int PAD_VALUE = 114;  // gray, the same padding Ultralytics trains with

// Where a frame lands inside a letterboxed model input.
struct Letterbox {
  double scale;  // applied to the frame
  int width, height;  // of the resized frame
  int pad_x, pad_y;  // padding before the resized frame
};

// Fit a frame_h x frame_w frame into an out_h x out_w input, keeping the aspect
// ratio, and center it. Sizes round half to even, as Python's round() does.
Letterbox letterbox_geometry(int frame_h, int frame_w, int out_h, int out_w);

// The frame resized and padded with PAD_VALUE gray to out_h x out_w.
cv::Mat letterbox(const cv::Mat& frame, const Letterbox& geometry, int out_h, int out_w);

// Model input for one frame, plus what's needed to map boxes back to the frame.
struct PreparedFrame {
  std::vector<float> blob;  // [1, 3, height, width] RGB in [0, 1]; empty when gpu_blob holds it
  int height, width;  // the model input's
  double scale;
  int pad_x, pad_y;
  int frame_height, frame_width;  // the original frame's
  // The same blob in GPU memory, from CUDA frame prep, for a GPU detector to
  // read where it is; null from CPU frame prep. Freed with the last copy.
  std::shared_ptr<const float> gpu_blob = nullptr;
};

class FramePrep {
 public:
  virtual ~FramePrep() = default;
  // A BGR frame (as OpenCV reads them) to model input.
  virtual PreparedFrame prepare(const cv::Mat& frame) = 0;
};

// Letterbox, BGR to RGB, HWC to CHW, and scale to [0, 1], on the CPU.
class CpuFramePrep : public FramePrep {
 public:
  CpuFramePrep(int height, int width) : height_(height), width_(width) {}
  PreparedFrame prepare(const cv::Mat& frame) override;

 private:
  int height_, width_;
};

// Where frames are prepared: CpuFramePrep above, or CudaFramePrep
// (cuda_frame_prep.hpp), which leaves the blob on the GPU.
enum class Prep { cpu, cuda };

// Whether this build has CUDA frame prep (CMake was given TensorRT_ROOT, and with it the CUDA toolkit).
bool cuda_prep_built();

// The frame prep for a height x width model input. Throws std::runtime_error
// for CUDA in a build without it.
std::unique_ptr<FramePrep> make_frame_prep(Prep prep, int height, int width);

}  // namespace tether
