#include "tether/assignment.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tether {

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

// The solver's working state, for nr <= nc. Names follow SciPy's source.
struct Solver {
  int nr, nc;
  const std::vector<double>& cost;  // row-major, nr x nc
  std::vector<double> u, v, shortest_path_costs;
  std::vector<int> path, col4row, row4col, remaining;
  std::vector<bool> sr, sc;

  Solver(int rows, int cols, const std::vector<double>& c)
      : nr(rows), nc(cols), cost(c), u(rows, 0.0), v(cols, 0.0), shortest_path_costs(cols),
        path(cols, -1), col4row(rows, -1), row4col(cols, -1), remaining(cols), sr(rows), sc(cols) {}

  // Shortest augmenting path from row i to a free column; returns that column.
  int augmenting_path(int i, double& min_val) {
    min_val = 0;
    int num_remaining = nc;
    // Columns are scanned in reverse order, as SciPy does; this decides ties.
    for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
    std::fill(sr.begin(), sr.end(), false);
    std::fill(sc.begin(), sc.end(), false);
    std::fill(shortest_path_costs.begin(), shortest_path_costs.end(), INF);

    int sink = -1;
    while (sink == -1) {
      int index = -1;
      double lowest = INF;
      sr[i] = true;
      for (int it = 0; it < num_remaining; ++it) {
        int j = remaining[it];
        double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
        if (r < shortest_path_costs[j]) {
          path[j] = i;
          shortest_path_costs[j] = r;
        }
        // Among equally short paths, prefer one ending at a free column.
        if (shortest_path_costs[j] < lowest || (shortest_path_costs[j] == lowest && row4col[j] == -1)) {
          lowest = shortest_path_costs[j];
          index = it;
        }
      }
      min_val = lowest;
      if (min_val == INF) return -1;  // infeasible; can't happen with finite costs

      int j = remaining[index];
      if (row4col[j] == -1)
        sink = j;
      else
        i = row4col[j];
      sc[j] = true;
      remaining[index] = remaining[--num_remaining];
    }
    return sink;
  }

  void solve() {
    for (int cur_row = 0; cur_row < nr; ++cur_row) {
      double min_val;
      int sink = augmenting_path(cur_row, min_val);
      if (sink < 0) throw std::invalid_argument("linear_sum_assignment: cost matrix is infeasible");

      // Update the dual variables.
      u[cur_row] += min_val;
      for (int i = 0; i < nr; ++i)
        if (sr[i] && i != cur_row) u[i] += min_val - shortest_path_costs[col4row[i]];
      for (int j = 0; j < nc; ++j)
        if (sc[j]) v[j] -= min_val - shortest_path_costs[j];

      // Augment the previous solution along the path.
      int j = sink;
      while (true) {
        int i = path[j];
        row4col[j] = i;
        std::swap(col4row[i], j);
        if (i == cur_row) break;
      }
    }
  }
};

}  // namespace

Pairs linear_sum_assignment(const CostMatrix& cost) {
  int nr = static_cast<int>(cost.size());
  int nc = nr ? static_cast<int>(cost[0].size()) : 0;
  if (nr == 0 || nc == 0) return {};
  for (const auto& row : cost)
    if (static_cast<int>(row.size()) != nc) throw std::invalid_argument("linear_sum_assignment: ragged matrix");

  // The solver needs at least as many columns as rows; SciPy transposes otherwise.
  bool transpose = nc < nr;
  if (transpose) std::swap(nr, nc);
  std::vector<double> flat(static_cast<size_t>(nr) * nc);
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nc; ++j) flat[static_cast<size_t>(i) * nc + j] = transpose ? cost[j][i] : cost[i][j];

  Solver solver(nr, nc, flat);
  solver.solve();

  Pairs pairs;
  if (transpose) {
    // Rows of the solved matrix are original columns; sort by original row.
    std::vector<int> order(nr);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return solver.col4row[a] < solver.col4row[b]; });
    for (int k : order) pairs.emplace_back(solver.col4row[k], k);
  } else {
    for (int i = 0; i < nr; ++i) pairs.emplace_back(i, solver.col4row[i]);
  }
  return pairs;
}

}  // namespace tether
