/**
 * @file runGCRODRTests.cpp
 * @brief Driver for GCRO-DR unit tests.
 */
#include "GCRODRTests.hpp"

int main()
{
  ReSolve::tests::GCRODRTests    test;
  ReSolve::tests::TestingResults result;

  result += test.initialCycleConverges();
  result += test.happyBreakdown();
  result += test.recycleSpaceProperties();
  result += test.recycledCyclesConverge();
  result += test.recycleSpaceIsUpdated();
  result += test.complexPairSelection();
  result += test.rightPreconditionedUpdateUsesTransformedBasis();
  result += test.systemSolverSequence();
  result += test.zeroInitialResidual();
  result += test.invalidSolveData();

  return result.summary();
}
