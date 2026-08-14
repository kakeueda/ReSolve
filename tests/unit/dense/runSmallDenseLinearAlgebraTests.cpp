/**
 * @file runSmallDenseLinearAlgebraTests.cpp
 * @brief Driver for small dense linear algebra tests.
 */
#include "SmallDenseLinearAlgebraTests.hpp"

int main()
{
  ReSolve::tests::SmallDenseLinearAlgebraTests test;
  ReSolve::tests::TestingResults               result;

  result += test.thinQr();
  result += test.triangularSolve();
  result += test.generalizedEigenvectors();
  result += test.complexGeneralizedEigenvectors();
  result += test.invalidArguments();

  return result.summary();
}
