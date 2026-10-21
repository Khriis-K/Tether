// The linear assignment problem: pick one cell per row and column, lowest total cost.
#pragma once

#include <utility>
#include <vector>

namespace tether {

using CostMatrix = std::vector<std::vector<double>>;  // rows of equal length
using Pairs = std::vector<std::pair<int, int>>;       // (row, column)

// Port of SciPy's linear_sum_assignment (Crouse's shortest augmenting path,
// scipy/optimize/rectangular_lsap). Visits rows and columns in the same order,
// so it breaks exact ties the way SciPy does. Assigns min(rows, columns) pairs,
// sorted by row. Costs must be finite.
Pairs linear_sum_assignment(const CostMatrix& cost);

}  // namespace tether
