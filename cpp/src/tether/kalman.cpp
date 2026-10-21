#include "tether/kalman.hpp"

#include <cmath>
#include <utility>

namespace tether {

namespace {

constexpr int N = 8;  // state size
constexpr int M = 4;  // measurement size: a detection gives position and size, not velocity

using Mat = std::array<std::array<double, N>, N>;

// A new track's center velocity is unknown, so it starts with a huge spread and
// the first matched detection sets it almost entirely on its own.
constexpr double INITIAL_VELOCITY_NOISE = 10.0;

// Diagonal covariance for the state, with each std scaled by box height h.
Mat covariance(double h, double center_velocity_noise) {
  const double stds[N] = {POSITION_NOISE,        POSITION_NOISE,        POSITION_NOISE,      POSITION_NOISE,
                          center_velocity_noise, center_velocity_noise, SIZE_VELOCITY_NOISE, SIZE_VELOCITY_NOISE};
  Mat p{};
  for (int i = 0; i < N; ++i) p[i][i] = (stds[i] * h) * (stds[i] * h);
  return p;
}

std::array<double, M> to_measurement(const Box& b) {
  return {(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]};
}

}  // namespace

BoxFilter::BoxFilter(const Box& box) : x_{}, p_{} {
  auto z = to_measurement(box);
  for (int i = 0; i < M; ++i) x_[i] = z[i];
  p_ = covariance(z[3], INITIAL_VELOCITY_NOISE);
}

Box BoxFilter::box() const {
  double cx = x_[0], cy = x_[1], w = x_[2], h = x_[3];
  return {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
}

void BoxFilter::predict(double damping) {
  // F = I + (position += velocity), with the velocities first scaled by damping:
  // F = [[I, d*I], [0, d*I]].
  Mat q = covariance(x_[3], VELOCITY_NOISE);
  for (int i = 0; i < M; ++i) {
    x_[i + M] *= damping;
    x_[i] += x_[i + M];
  }
  // P = F P F^T + Q. F acts on rows (F P), then on columns (... F^T).
  Mat fp = p_;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i) {
      fp[i + M][j] = damping * p_[i + M][j];
      fp[i][j] = p_[i][j] + fp[i + M][j];
    }
  Mat fpf = fp;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) {
      fpf[i][j + M] = damping * fp[i][j + M];
      fpf[i][j] = fp[i][j] + fpf[i][j + M];
    }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) p_[i][j] = fpf[i][j] + q[i][j];
}

void BoxFilter::update(const Box& box) {
  auto z = to_measurement(box);
  double r = POSITION_NOISE * x_[3];
  r *= r;

  // S = H P H^T + R: the top-left 4x4 of P plus R. B = H P: the top 4 rows of P.
  double s[M][M];
  double b[M][N];
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) s[i][j] = p_[i][j] + (i == j ? r : 0.0);
    for (int j = 0; j < N; ++j) b[i][j] = p_[i][j];
  }
  // Solve S X = B by Gaussian elimination with partial pivoting; the gain is
  // K = X^T = P H^T S^-1, as S and P are symmetric.
  for (int col = 0; col < M; ++col) {
    int pivot = col;
    for (int row = col + 1; row < M; ++row)
      if (std::abs(s[row][col]) > std::abs(s[pivot][col])) pivot = row;
    if (pivot != col) {
      std::swap(s[pivot], s[col]);
      std::swap(b[pivot], b[col]);
    }
    for (int row = col + 1; row < M; ++row) {
      double f = s[row][col] / s[col][col];
      for (int k = col; k < M; ++k) s[row][k] -= f * s[col][k];
      for (int k = 0; k < N; ++k) b[row][k] -= f * b[col][k];
    }
  }
  for (int row = M - 1; row >= 0; --row) {
    for (int k = row + 1; k < M; ++k)
      for (int j = 0; j < N; ++j) b[row][j] -= s[row][k] * b[k][j];
    for (int j = 0; j < N; ++j) b[row][j] /= s[row][row];
  }
  // b now holds X = S^-1 H P, so the gain is K[i][j] = b[j][i].

  double innovation[M];
  for (int j = 0; j < M; ++j) innovation[j] = z[j] - x_[j];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j) x_[i] += b[j][i] * innovation[j];

  // P = (I - K H) P = P - K (H P), where H P is the top 4 rows of P.
  Mat next = p_;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < M; ++k) next[i][j] -= b[k][i] * p_[k][j];
  p_ = next;
}

}  // namespace tether
