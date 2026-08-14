/**
 * @file SmallDenseLinearAlgebra.cpp
 * @brief Implementation of host LAPACK helpers for small dense problems.
 */
#include "SmallDenseLinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C"
{
  void sgeqrf_(const int*, const int*, float*, const int*, float*, float*, const int*, int*);
  void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);

  void sorgqr_(const int*, const int*, const int*, float*, const int*, const float*, float*, const int*, int*);
  void dorgqr_(const int*, const int*, const int*, double*, const int*, const double*, double*, const int*, int*);

  void strtrs_(const char*, const char*, const char*, const int*, const int*, const float*, const int*, float*, const int*, int*);
  void dtrtrs_(const char*, const char*, const char*, const int*, const int*, const double*, const int*, double*, const int*, int*);

  void sggev_(const char*,
              const char*,
              const int*,
              float*,
              const int*,
              float*,
              const int*,
              float*,
              float*,
              float*,
              float*,
              const int*,
              float*,
              const int*,
              float*,
              const int*,
              int*);
  void dggev_(const char*,
              const char*,
              const int*,
              double*,
              const int*,
              double*,
              const int*,
              double*,
              double*,
              double*,
              double*,
              const int*,
              double*,
              const int*,
              double*,
              const int*,
              int*);
}

namespace ReSolve
{
  namespace
  {
    template <typename T>
    struct Lapack;

    template <>
    struct Lapack<float>
    {
      static void geqrf(const int* m,
                        const int* n,
                        float*     A,
                        const int* lda,
                        float*     tau,
                        float*     work,
                        const int* lwork,
                        int*       info)
      {
        sgeqrf_(m, n, A, lda, tau, work, lwork, info);
      }

      static void orgqr(const int* m,
                        const int* n,
                        const int* k,
                        float*     A,
                        const int* lda,
                        float*     tau,
                        float*     work,
                        const int* lwork,
                        int*       info)
      {
        sorgqr_(m, n, k, A, lda, tau, work, lwork, info);
      }

      static void trtrs(const char*  uplo,
                        const char*  trans,
                        const char*  diag,
                        const int*   n,
                        const int*   nrhs,
                        const float* A,
                        const int*   lda,
                        float*       B,
                        const int*   ldb,
                        int*         info)
      {
        strtrs_(uplo, trans, diag, n, nrhs, A, lda, B, ldb, info);
      }

      static void ggev(const char* jobvl,
                       const char* jobvr,
                       const int*  n,
                       float*      A,
                       const int*  lda,
                       float*      B,
                       const int*  ldb,
                       float*      alphar,
                       float*      alphai,
                       float*      beta,
                       float*      VL,
                       const int*  ldvl,
                       float*      VR,
                       const int*  ldvr,
                       float*      work,
                       const int*  lwork,
                       int*        info)
      {
        sggev_(jobvl,
               jobvr,
               n,
               A,
               lda,
               B,
               ldb,
               alphar,
               alphai,
               beta,
               VL,
               ldvl,
               VR,
               ldvr,
               work,
               lwork,
               info);
      }
    };

    template <>
    struct Lapack<double>
    {
      static void geqrf(const int* m,
                        const int* n,
                        double*    A,
                        const int* lda,
                        double*    tau,
                        double*    work,
                        const int* lwork,
                        int*       info)
      {
        dgeqrf_(m, n, A, lda, tau, work, lwork, info);
      }

      static void orgqr(const int* m,
                        const int* n,
                        const int* k,
                        double*    A,
                        const int* lda,
                        double*    tau,
                        double*    work,
                        const int* lwork,
                        int*       info)
      {
        dorgqr_(m, n, k, A, lda, tau, work, lwork, info);
      }

      static void trtrs(const char*   uplo,
                        const char*   trans,
                        const char*   diag,
                        const int*    n,
                        const int*    nrhs,
                        const double* A,
                        const int*    lda,
                        double*       B,
                        const int*    ldb,
                        int*          info)
      {
        dtrtrs_(uplo, trans, diag, n, nrhs, A, lda, B, ldb, info);
      }

      static void ggev(const char* jobvl,
                       const char* jobvr,
                       const int*  n,
                       double*     A,
                       const int*  lda,
                       double*     B,
                       const int*  ldb,
                       double*     alphar,
                       double*     alphai,
                       double*     beta,
                       double*     VL,
                       const int*  ldvl,
                       double*     VR,
                       const int*  ldvr,
                       double*     work,
                       const int*  lwork,
                       int*        info)
      {
        dggev_(jobvl,
               jobvr,
               n,
               A,
               lda,
               B,
               ldb,
               alphar,
               alphai,
               beta,
               VL,
               ldvl,
               VR,
               ldvr,
               work,
               lwork,
               info);
      }
    };

    bool toLapackInt(index_type value, int& result)
    {
      const long double converted = static_cast<long double>(value);
      if (converted < 0.0L || converted > static_cast<long double>(std::numeric_limits<int>::max()))
      {
        return false;
      }
      result = static_cast<int>(value);
      return true;
    }

    bool isValidTriangularOption(char value, char first, char second, char third = '\0')
    {
      return value == first || value == second || (third != '\0' && value == third);
    }
  } // namespace

  int SmallDenseLinearAlgebra::thinQr(index_type m,
                                      index_type n,
                                      real_type* A,
                                      index_type lda,
                                      real_type* R,
                                      index_type ldr)
  {
    int m_lapack   = 0;
    int n_lapack   = 0;
    int lda_lapack = 0;
    int ldr_lapack = 0;
    if (!toLapackInt(m, m_lapack) || !toLapackInt(n, n_lapack)
        || !toLapackInt(lda, lda_lapack) || !toLapackInt(ldr, ldr_lapack)
        || m_lapack < n_lapack || lda_lapack < std::max(1, m_lapack)
        || ldr_lapack < std::max(1, n_lapack) || (n_lapack > 0 && (A == nullptr || R == nullptr)))
    {
      return -1;
    }
    if (n_lapack == 0)
    {
      return 0;
    }

    std::vector<real_type> tau(static_cast<std::size_t>(n_lapack));
    real_type              work_query = 0.0;
    int                    lwork      = -1;
    int                    info       = 0;
    Lapack<real_type>::geqrf(&m_lapack,
                             &n_lapack,
                             A,
                             &lda_lapack,
                             tau.data(),
                             &work_query,
                             &lwork,
                             &info);
    if (info != 0)
    {
      return info;
    }

    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    std::vector<real_type> work(static_cast<std::size_t>(lwork));
    Lapack<real_type>::geqrf(&m_lapack,
                             &n_lapack,
                             A,
                             &lda_lapack,
                             tau.data(),
                             work.data(),
                             &lwork,
                             &info);
    if (info != 0)
    {
      return info;
    }

    for (int j = 0; j < n_lapack; ++j)
    {
      for (int i = 0; i < n_lapack; ++i)
      {
        R[j * ldr_lapack + i] = (i <= j) ? A[j * lda_lapack + i] : real_type{0.0};
      }
    }

    work_query = 0.0;
    lwork      = -1;
    Lapack<real_type>::orgqr(&m_lapack,
                             &n_lapack,
                             &n_lapack,
                             A,
                             &lda_lapack,
                             tau.data(),
                             &work_query,
                             &lwork,
                             &info);
    if (info != 0)
    {
      return info;
    }

    lwork = std::max(1, static_cast<int>(std::ceil(work_query)));
    work.resize(static_cast<std::size_t>(lwork));
    Lapack<real_type>::orgqr(&m_lapack,
                             &n_lapack,
                             &n_lapack,
                             A,
                             &lda_lapack,
                             tau.data(),
                             work.data(),
                             &lwork,
                             &info);
    return info;
  }

  int SmallDenseLinearAlgebra::triangularSolve(char             uplo,
                                               char             trans,
                                               char             diag,
                                               index_type       n,
                                               index_type       nrhs,
                                               const real_type* A,
                                               index_type       lda,
                                               real_type*       B,
                                               index_type       ldb)
  {
    int n_lapack    = 0;
    int nrhs_lapack = 0;
    int lda_lapack  = 0;
    int ldb_lapack  = 0;
    if (!toLapackInt(n, n_lapack) || !toLapackInt(nrhs, nrhs_lapack)
        || !toLapackInt(lda, lda_lapack) || !toLapackInt(ldb, ldb_lapack)
        || !isValidTriangularOption(uplo, 'U', 'L')
        || !isValidTriangularOption(trans, 'N', 'T', 'C')
        || !isValidTriangularOption(diag, 'N', 'U')
        || lda_lapack < std::max(1, n_lapack) || ldb_lapack < std::max(1, n_lapack)
        || (n_lapack > 0 && nrhs_lapack > 0 && (A == nullptr || B == nullptr)))
    {
      return -1;
    }
    if (n_lapack == 0 || nrhs_lapack == 0)
    {
      return 0;
    }

    int info = 0;
    Lapack<real_type>::trtrs(&uplo,
                             &trans,
                             &diag,
                             &n_lapack,
                             &nrhs_lapack,
                             A,
                             &lda_lapack,
                             B,
                             &ldb_lapack,
                             &info);
    return info;
  }

  int SmallDenseLinearAlgebra::generalizedEigenvectors(index_type n,
                                                       real_type* A,
                                                       index_type lda,
                                                       real_type* B,
                                                       index_type ldb,
                                                       real_type* alphar,
                                                       real_type* alphai,
                                                       real_type* beta,
                                                       real_type* VR,
                                                       index_type ldvr)
  {
    int n_lapack    = 0;
    int lda_lapack  = 0;
    int ldb_lapack  = 0;
    int ldvr_lapack = 0;
    if (!toLapackInt(n, n_lapack) || !toLapackInt(lda, lda_lapack)
        || !toLapackInt(ldb, ldb_lapack) || !toLapackInt(ldvr, ldvr_lapack)
        || lda_lapack < std::max(1, n_lapack) || ldb_lapack < std::max(1, n_lapack)
        || ldvr_lapack < std::max(1, n_lapack)
        || (n_lapack > 0
            && (A == nullptr || B == nullptr || alphar == nullptr || alphai == nullptr
                || beta == nullptr || VR == nullptr)))
    {
      return -1;
    }
    if (n_lapack == 0)
    {
      return 0;
    }

    const char jobvl      = 'N';
    const char jobvr      = 'V';
    int        ldvl       = 1;
    real_type  vl         = 0.0;
    real_type  work_query = 0.0;
    int        lwork      = -1;
    int        info       = 0;

    Lapack<real_type>::ggev(&jobvl,
                            &jobvr,
                            &n_lapack,
                            A,
                            &lda_lapack,
                            B,
                            &ldb_lapack,
                            alphar,
                            alphai,
                            beta,
                            &vl,
                            &ldvl,
                            VR,
                            &ldvr_lapack,
                            &work_query,
                            &lwork,
                            &info);
    if (info != 0)
    {
      return info;
    }

    lwork = std::max(std::max(1, 8 * n_lapack),
                     static_cast<int>(std::ceil(work_query)));
    std::vector<real_type> work(static_cast<std::size_t>(lwork));
    Lapack<real_type>::ggev(&jobvl,
                            &jobvr,
                            &n_lapack,
                            A,
                            &lda_lapack,
                            B,
                            &ldb_lapack,
                            alphar,
                            alphai,
                            beta,
                            &vl,
                            &ldvl,
                            VR,
                            &ldvr_lapack,
                            work.data(),
                            &lwork,
                            &info);
    return info;
  }
} // namespace ReSolve
