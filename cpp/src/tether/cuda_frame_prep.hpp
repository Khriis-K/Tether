// CUDA frame prep: CpuFramePrep's letterbox, BGR to RGB, HWC to CHW and scale
// to [0, 1] as one CUDA kernel, leaving the blob on the GPU for TensorRT to read
// where it is. Only built when CMake is given TensorRT_ROOT (and so the CUDA toolkit).
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "tether/frame_prep.hpp"

namespace tether {

// Each prepare uploads the 8-bit frame (smaller than the float blob it becomes
// once the input is bigger than about 1280x540) and runs the kernel on this
// prep's own CUDA stream, returning once the blob is ready. The blob matches
// CpuFramePrep's within one 8-bit gray level per value (the resize rounds in
// fixed point as OpenCV's does, but can round a pixel the other way).
class CudaFramePrep : public FramePrep {
 public:
  // Throws std::runtime_error when there's no CUDA GPU to run on.
  CudaFramePrep(int height, int width);
  ~CudaFramePrep() override;
  // The returned frame's gpu_blob holds the blob; its blob is empty. Each
  // frame gets its own GPU buffer, held until its last copy is gone, so a
  // frame can be prepared while the detector still reads the one before.
  PreparedFrame prepare(const cv::Mat& frame) override;

 private:
  struct Gpu;
  std::unique_ptr<Gpu> gpu_;
  int height_, width_;
};

// A prepared frame's blob on the CPU, wherever it was prepared: for checks and
// tests, as it copies the GPU blob back.
std::vector<float> download_blob(const PreparedFrame& prepared);

}  // namespace tether
