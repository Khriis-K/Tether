#include "tether/cuda_frame_prep.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tether {

namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

// OpenCV's 8-bit bilinear resize weighs its two source pixels in fixed point,
// out of 2048 (INTER_RESIZE_COEF_BITS = 11), and rounds once after both passes.
constexpr int WEIGHT_BITS = 11;
constexpr float WEIGHT_SCALE = 1 << WEIGHT_BITS;

// One output pixel's two source pixels along one axis, and their weights.
struct Tap {
  int first;  // the second is first + 1, or first again at the edge
  short w0, w1;
};

// The taps for resizing `src` pixels to `dst`, computed as cv::resize does
// for INTER_LINEAR (resize.cpp): half-pixel centers, clamped at both edges,
// each weight rounded to the nearest 1/2048 on its own (halves to even, as
// saturate_cast<short> does).
std::vector<Tap> taps(int src, int dst) {
  double scale = 1.0 / (static_cast<double>(dst) / src);  // OpenCV's 1 / inv_scale, not src / dst
  std::vector<Tap> out(dst);
  for (int d = 0; d < dst; ++d) {
    float f = static_cast<float>((d + 0.5) * scale - 0.5);
    int s = static_cast<int>(std::floor(f));
    f -= static_cast<float>(s);
    if (s < 0) f = 0, s = 0;
    if (s >= src - 1) f = 0, s = src - 1;
    out[d] = {s, static_cast<short>(std::lrint((1.0f - f) * WEIGHT_SCALE)),
              static_cast<short>(std::lrint(f * WEIGHT_SCALE))};
  }
  return out;
}

// One thread per model-input pixel: letterbox (bilinear resize, PAD_VALUE
// gray around it), BGR to RGB, HWC to CHW and scale to [0, 1], as CpuFramePrep.
__global__ void prep_kernel(const unsigned char* frame, int frame_pitch, int frame_w, int frame_h, const Tap* x_taps,
                            const Tap* y_taps, int pad_x, int pad_y, int resized_w, int resized_h, float* blob,
                            int out_w, int out_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;
  int bgr[3] = {PAD_VALUE, PAD_VALUE, PAD_VALUE};
  int rx = x - pad_x, ry = y - pad_y;
  if (rx >= 0 && rx < resized_w && ry >= 0 && ry < resized_h) {
    Tap tx = x_taps[rx], ty = y_taps[ry];
    int x0 = tx.first * 3, x1 = min(tx.first + 1, frame_w - 1) * 3;
    const unsigned char* row0 = frame + static_cast<size_t>(ty.first) * frame_pitch;
    const unsigned char* row1 = frame + static_cast<size_t>(min(ty.first + 1, frame_h - 1)) * frame_pitch;
    for (int c = 0; c < 3; ++c) {
      // Across, then down, as OpenCV's HResizeLinear and its vectorized
      // VResizeLinear (VResizeLinearVec_32s8u), which drops 4 bits, keeps the
      // high 16 of each product, and rounds off the last 2. Its scalar form,
      // one rounding of the exact sum, differs on about one value in ten.
      int top = row0[x0 + c] * tx.w0 + row0[x1 + c] * tx.w1;
      int bottom = row1[x0 + c] * tx.w0 + row1[x1 + c] * tx.w1;
      int v = (((top >> 4) * ty.w0) >> 16) + (((bottom >> 4) * ty.w1) >> 16);
      v = (v + 2) >> 2;
      bgr[c] = min(max(v, 0), 255);
    }
  }
  size_t plane = static_cast<size_t>(out_h) * out_w, at = static_cast<size_t>(y) * out_w + x;
  // Divides in float, as CpuFramePrep does (nvcc's default '/' is correctly rounded).
  blob[at] = static_cast<float>(bgr[2]) / 255.0f;
  blob[plane + at] = static_cast<float>(bgr[1]) / 255.0f;
  blob[2 * plane + at] = static_cast<float>(bgr[0]) / 255.0f;
}

// GPU blobs of one size, reused: cudaFree waits for the whole GPU, so freeing
// one each frame would stall the detector. Kept alive by every blob it lent.
class BlobPool {
 public:
  explicit BlobPool(size_t elements) : elements_(elements) {}
  ~BlobPool() {
    for (float* p : free_) cudaFree(p);
  }
  std::shared_ptr<const float> take(const std::shared_ptr<BlobPool>& self) {
    float* p = nullptr;
    {
      std::lock_guard lock(mutex_);
      if (!free_.empty()) {
        p = free_.back();
        free_.pop_back();
      }
    }
    if (!p) check(cudaMalloc(&p, elements_ * sizeof(float)), "cudaMalloc");
    return {p, [self](const float* q) { self->give(const_cast<float*>(q)); }};
  }

 private:
  void give(float* p) {
    std::lock_guard lock(mutex_);
    free_.push_back(p);
  }
  size_t elements_;
  std::mutex mutex_;  // blobs come back on whichever thread drops them last
  std::vector<float*> free_;
};

}  // namespace

struct CudaFramePrep::Gpu {
  cudaStream_t stream = nullptr;
  unsigned char* frame = nullptr;  // the uploaded 8-bit frame, rows back to back
  size_t frame_bytes = 0;
  Tap* gpu_taps = nullptr;  // x taps, then y taps, for frame_size
  cv::Size frame_size;
  Letterbox geometry{};
  std::shared_ptr<BlobPool> pool;

  ~Gpu() {
    cudaFree(frame);
    cudaFree(gpu_taps);
    if (stream) cudaStreamDestroy(stream);
  }

  // The taps and geometry for frames of `size`, made again only when it changes.
  void fit(cv::Size size, int out_h, int out_w) {
    if (size == frame_size) return;
    geometry = letterbox_geometry(size.height, size.width, out_h, out_w);
    auto x = taps(size.width, geometry.width), y = taps(size.height, geometry.height);
    x.insert(x.end(), y.begin(), y.end());
    cudaFree(gpu_taps);
    gpu_taps = nullptr;
    frame_size = {};
    check(cudaMalloc(&gpu_taps, x.size() * sizeof(Tap)), "cudaMalloc");
    check(cudaMemcpy(gpu_taps, x.data(), x.size() * sizeof(Tap), cudaMemcpyHostToDevice), "copy to GPU");
    size_t bytes = static_cast<size_t>(size.width) * size.height * 3;
    if (bytes > frame_bytes) {
      cudaFree(frame);
      frame = nullptr;
      frame_bytes = 0;
      check(cudaMalloc(&frame, bytes), "cudaMalloc");
      frame_bytes = bytes;
    }
    frame_size = size;
  }
};

CudaFramePrep::CudaFramePrep(int height, int width) : gpu_(std::make_unique<Gpu>()), height_(height), width_(width) {
  check(cudaStreamCreate(&gpu_->stream), "no CUDA GPU for CUDA frame prep (cudaStreamCreate)");
  gpu_->pool = std::make_shared<BlobPool>(3 * static_cast<size_t>(height) * width);
}

CudaFramePrep::~CudaFramePrep() = default;

PreparedFrame CudaFramePrep::prepare(const cv::Mat& frame) {
  if (frame.type() != CV_8UC3) throw std::invalid_argument("CUDA frame prep takes 8-bit BGR frames");
  auto& g = *gpu_;
  g.fit(frame.size(), height_, width_);
  const Letterbox& lb = g.geometry;
  int row_bytes = frame.cols * 3;
  check(cudaMemcpy2DAsync(g.frame, row_bytes, frame.data, frame.step, row_bytes, frame.rows, cudaMemcpyHostToDevice,
                          g.stream),
        "copy to GPU");
  auto blob = g.pool->take(g.pool);
  dim3 block(32, 8), grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
  prep_kernel<<<grid, block, 0, g.stream>>>(g.frame, row_bytes, frame.cols, frame.rows, g.gpu_taps, g.gpu_taps + lb.width,
                                            lb.pad_x, lb.pad_y, lb.width, lb.height, const_cast<float*>(blob.get()),
                                            width_, height_);
  check(cudaGetLastError(), "frame prep kernel");
  // Done when it returns: the step's time is the whole prep, and any stream may read the blob.
  check(cudaStreamSynchronize(g.stream), "cudaStreamSynchronize");
  return {{}, height_, width_, lb.scale, lb.pad_x, lb.pad_y, frame.rows, frame.cols, std::move(blob)};
}

std::vector<float> download_blob(const PreparedFrame& prepared) {
  if (!prepared.gpu_blob) return prepared.blob;
  std::vector<float> out(3 * static_cast<size_t>(prepared.height) * prepared.width);
  check(cudaMemcpy(out.data(), prepared.gpu_blob.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "copy from GPU");
  return out;
}

}  // namespace tether
