#include "tether/frame_prep.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#ifdef TETHER_HAS_CUDA_PREP
#include "tether/cuda_frame_prep.hpp"
#endif

namespace tether {

Letterbox letterbox_geometry(int frame_h, int frame_w, int out_h, int out_w) {
  double scale = std::min(static_cast<double>(out_h) / frame_h, static_cast<double>(out_w) / frame_w);
  // std::nearbyint rounds half to even (the default rounding mode), as Python's round() does.
  int width = static_cast<int>(std::nearbyint(frame_w * scale));
  int height = static_cast<int>(std::nearbyint(frame_h * scale));
  return {scale, width, height, (out_w - width) / 2, (out_h - height) / 2};
}

cv::Mat letterbox(const cv::Mat& frame, const Letterbox& g, int out_h, int out_w) {
  cv::Mat image(out_h, out_w, CV_8UC3, cv::Scalar::all(PAD_VALUE));
  cv::Mat target = image(cv::Rect(g.pad_x, g.pad_y, g.width, g.height));
  if (frame.cols == g.width && frame.rows == g.height)
    frame.copyTo(target);
  else
    cv::resize(frame, target, target.size(), 0, 0, cv::INTER_LINEAR);
  return image;
}

PreparedFrame CpuFramePrep::prepare(const cv::Mat& frame) {
  auto g = letterbox_geometry(frame.rows, frame.cols, height_, width_);
  cv::Mat image = letterbox(frame, g, height_, width_);

  // BGR HWC uint8 -> RGB CHW float in [0, 1]. Divides in float, as numpy does,
  // so the blob is bit-identical to the Python one.
  PreparedFrame out{std::vector<float>(3 * static_cast<size_t>(height_) * width_), height_, width_, g.scale,
                    g.pad_x, g.pad_y, frame.rows, frame.cols};
  size_t plane = static_cast<size_t>(height_) * width_;
  float* r = out.blob.data();
  float* gr = r + plane;
  float* b = gr + plane;
  for (int y = 0; y < height_; ++y) {
    const auto* px = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width_; ++x, ++r, ++gr, ++b) {
      *b = static_cast<float>(px[x][0]) / 255.0f;
      *gr = static_cast<float>(px[x][1]) / 255.0f;
      *r = static_cast<float>(px[x][2]) / 255.0f;
    }
  }
  return out;
}

bool cuda_prep_built() {
#ifdef TETHER_HAS_CUDA_PREP
  return true;
#else
  return false;
#endif
}

std::unique_ptr<FramePrep> make_frame_prep(Prep prep, int height, int width) {
  if (prep == Prep::cpu) return std::make_unique<CpuFramePrep>(height, width);
#ifdef TETHER_HAS_CUDA_PREP
  return std::make_unique<CudaFramePrep>(height, width);
#else
  throw std::runtime_error("this build has no CUDA frame prep; configure CMake with -DTensorRT_ROOT=<TensorRT folder>, "
                           "which brings in the CUDA toolkit");
#endif
}

}  // namespace tether
