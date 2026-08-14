/**
 * @file SmallDenseLinearAlgebra.hpp
 * @brief Host LAPACK helpers for small dense problems used by Krylov solvers.
 */
#pragma once

#include <resolve/Common.hpp>

namespace ReSolve
{
  /**
   * @brief Small dense linear algebra operations performed in host memory.
   *
   * All matrices use LAPACK-compatible column-major storage. This class keeps
   * the small dense operations needed by GCRO-DR independent of the CPU, CUDA,
   * and HIP backends used for sparse matrices and large vectors.
   */
  class SmallDenseLinearAlgebra
  {
  public:
    /**
     * @brief Compute a thin QR factorization of an m-by-n matrix, m >= n.
     *
     * On return, A contains the m-by-n matrix Q and R contains the n-by-n
     * upper-triangular factor. The input matrix A is overwritten.
     */
    static int thinQr(index_type m,
                      index_type n,
                      real_type* A,
                      index_type lda,
                      real_type* R,
                      index_type ldr);

    /**
     * @brief Solve op(A) X = B for triangular A.
     *
     * A and B use column-major storage. B is overwritten with X.
     */
    static int triangularSolve(char             uplo,
                               char             trans,
                               char             diag,
                               index_type       n,
                               index_type       nrhs,
                               const real_type* A,
                               index_type       lda,
                               real_type*       B,
                               index_type       ldb);

    /**
     * @brief Solve A v = lambda B v for right generalized eigenvectors.
     *
     * A and B are overwritten. The generalized eigenvalues are represented as
     * (alphar + i*alphai)/beta. For a complex conjugate pair, LAPACK stores the
     * real and imaginary parts of the eigenvector in adjacent columns of VR.
     */
    static int generalizedEigenvectors(index_type n,
                                       real_type* A,
                                       index_type lda,
                                       real_type* B,
                                       index_type ldb,
                                       real_type* alphar,
                                       real_type* alphai,
                                       real_type* beta,
                                       real_type* VR,
                                       index_type ldvr);

  private:
    SmallDenseLinearAlgebra() = delete;
  };
} // namespace ReSolve
