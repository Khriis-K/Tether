// The assignment solver must pick exactly what SciPy's linear_sum_assignment
// picks, ties included, or the two trackers' IDs drift apart (docs/FORMATS.md).
// Expected pairs below were printed by SciPy 1.16.3 for the same matrices.

#include <gtest/gtest.h>

#include <vector>

#include "tether/assignment.hpp"

using tether::CostMatrix;
using tether::Pairs;

struct Case {
  CostMatrix cost;
  Pairs expected;
};

TEST(Assignment, MatchesScipyIncludingTiesAndRectangularShapes) {
  std::vector<Case> cases = {
      {{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}}, {{0, 0}, {1, 1}, {2, 2}}},
      {{{1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}, {{0, 0}, {1, 1}}},
      {{{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}}, {{0, 0}, {1, 1}}},
      {{{1.0, 1000000.0, 1.0}, {1000000.0, 1.0, 1.0}, {1.0, 1.0, 1000000.0}}, {{0, 0}, {1, 2}, {2, 1}}},
      {{{0.5, 0.5}, {0.5, 0.5}, {0.2, 0.2}}, {{1, 0}, {2, 1}}},
      {{{2.0, 0.0, 0.0, 0.0, 0.0}, {2.0, 2.0, 1.0, 0.0, 0.0}, {0.0, 1.0, 1.0, 1.0, 0.0}, {0.0, 2.0, 2.0, 0.0, 0.0}},
       {{0, 1}, {1, 3}, {2, 0}, {3, 4}}},
      {{{1.0, 1.0, 2.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 0.0}, {2.0, 2.0, 2.0}, {2.0, 0.0, 0.0}},
       {{0, 0}, {2, 2}, {4, 1}}},
      {{{0.6962159966701554, 0.2927207490124871, 0.0014900835088361708, 0.9734602747664127},
        {0.29840122301687566, 0.3139860020343368, 0.8917110704451572, 0.5851629398909081},
        {0.47130966518183137, 0.7732770096488164, 0.030346007662471197, 0.7069650956556235},
        {0.3742438334784708, 0.09085271350425783, 0.6605000674278948, 0.9314638547413545},
        {0.20719116808100124, 0.630090199785343, 0.29816309065742475, 0.7417566800693304},
        {0.7221648081421175, 0.21871542456880455, 0.8298868742743123, 0.6576522108732432}},
       {{0, 2}, {1, 3}, {3, 1}, {4, 0}}},
  };

  for (size_t i = 0; i < cases.size(); ++i)
    EXPECT_EQ(tether::linear_sum_assignment(cases[i].cost), cases[i].expected) << "case " << i;
}

TEST(Assignment, EmptyMatrixHasNoPairs) {
  EXPECT_EQ(tether::linear_sum_assignment({}), Pairs{});
  EXPECT_EQ(tether::linear_sum_assignment({{}, {}}), Pairs{});
}
