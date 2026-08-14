/**
 * @file SmallDenseLinearAlgebraTests.hpp
 * @brief Unit tests for small dense host LAPACK operations.
 */
#pragma once

#include <algorithm>
#include <cmath>

#include <resolve/SmallDenseLinearAlgebra.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    class SmallDenseLinearAlgebraTests : TestBase
    {
    public:
      TestOutcome thinQr()
      {
        const index_type m            = 3;
        const index_type n            = 2;
        real_type        A[]          = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
        const real_type  A_original[] = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
        real_type        R[4]         = {0.0, 0.0, 0.0, 0.0};

        TestStatus success;
        success = SmallDenseLinearAlgebra::thinQr(m, n, A, m, R, n) == 0;

        const real_type tolerance = 100.0 * constants::MACHINE_EPSILON;
        for (index_type j = 0; j < n; ++j)
        {
          for (index_type i = 0; i < n; ++i)
          {
            real_type dot = 0.0;
            for (index_type row = 0; row < m; ++row)
            {
              dot += A[i * m + row] * A[j * m + row];
            }
            const real_type expected = i == j ? 1.0 : 0.0;
            success *= std::abs(dot - expected) <= tolerance;
          }
        }

        for (index_type j = 0; j < n; ++j)
        {
          for (index_type row = 0; row < m; ++row)
          {
            real_type reconstructed = 0.0;
            for (index_type k = 0; k < n; ++k)
            {
              reconstructed += A[k * m + row] * R[j * n + k];
            }
            success *= std::abs(reconstructed - A_original[j * m + row]) <= tolerance;
          }
        }

        return success.report(__func__);
      }

      TestOutcome triangularSolve()
      {
        const real_type A[] = {2.0, 0.0, 1.0, 3.0};
        real_type       B[] = {5.0, 6.0};

        TestStatus success;
        success = SmallDenseLinearAlgebra::triangularSolve('U',
                                                           'N',
                                                           'N',
                                                           2,
                                                           1,
                                                           A,
                                                           2,
                                                           B,
                                                           2)
                  == 0;
        const real_type tolerance = 20.0 * constants::MACHINE_EPSILON;
        success *= std::abs(B[0] - 1.5) <= tolerance;
        success *= std::abs(B[1] - 2.0) <= tolerance;
        return success.report(__func__);
      }

      TestOutcome generalizedEigenvectors()
      {
        real_type A[]       = {2.0, 0.0, 0.0, 3.0};
        real_type B[]       = {1.0, 0.0, 0.0, 1.0};
        real_type alphar[2] = {0.0, 0.0};
        real_type alphai[2] = {0.0, 0.0};
        real_type beta[2]   = {0.0, 0.0};
        real_type VR[4]     = {0.0, 0.0, 0.0, 0.0};

        TestStatus success;
        success = SmallDenseLinearAlgebra::generalizedEigenvectors(2,
                                                                   A,
                                                                   2,
                                                                   B,
                                                                   2,
                                                                   alphar,
                                                                   alphai,
                                                                   beta,
                                                                   VR,
                                                                   2)
                  == 0;

        real_type eigenvalues[2] = {alphar[0] / beta[0], alphar[1] / beta[1]};
        std::sort(eigenvalues, eigenvalues + 2);
        const real_type tolerance = 50.0 * constants::MACHINE_EPSILON;
        success *= std::abs(alphai[0]) <= tolerance;
        success *= std::abs(alphai[1]) <= tolerance;
        success *= std::abs(eigenvalues[0] - 2.0) <= tolerance;
        success *= std::abs(eigenvalues[1] - 3.0) <= tolerance;
        return success.report(__func__);
      }

      TestOutcome complexGeneralizedEigenvectors()
      {
        // Rotation through 90 degrees has eigenvalues +i and -i.
        real_type A[]      = {0.0, 1.0, -1.0, 0.0};
        real_type B[]      = {1.0, 0.0, 0.0, 1.0};
        real_type alphar[] = {0.0, 0.0};
        real_type alphai[] = {0.0, 0.0};
        real_type beta[]   = {0.0, 0.0};
        real_type VR[]     = {0.0, 0.0, 0.0, 0.0};

        TestStatus success;
        success = SmallDenseLinearAlgebra::generalizedEigenvectors(2,
                                                                   A,
                                                                   2,
                                                                   B,
                                                                   2,
                                                                   alphar,
                                                                   alphai,
                                                                   beta,
                                                                   VR,
                                                                   2)
                  == 0;

        const real_type tolerance = 50.0 * constants::MACHINE_EPSILON;
        success *= std::abs(alphar[0]) <= tolerance;
        success *= std::abs(alphar[1]) <= tolerance;
        success *= std::abs(std::abs(alphai[0] / beta[0]) - 1.0) <= tolerance;
        success *= std::abs(std::abs(alphai[1] / beta[1]) - 1.0) <= tolerance;
        success *= alphai[0] * alphai[1] < 0.0;
        return success.report(__func__);
      }

      TestOutcome invalidArguments()
      {
        real_type  value = 0.0;
        TestStatus success;
        success = SmallDenseLinearAlgebra::thinQr(1, 2, &value, 1, &value, 1) != 0;
        success *= SmallDenseLinearAlgebra::triangularSolve('X',
                                                            'N',
                                                            'N',
                                                            1,
                                                            1,
                                                            &value,
                                                            1,
                                                            &value,
                                                            1)
                   != 0;
        return success.report(__func__);
      }
    };
  } // namespace tests
} // namespace ReSolve
