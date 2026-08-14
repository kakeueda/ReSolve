/**
 * @file LinSolverIterativeGCRODR.cpp
 * @brief Fixed right-preconditioned GCRO-DR implementation.
 */
#include "LinSolverIterativeGCRODR.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <resolve/GramSchmidt.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/Profiling.hpp>
#include <resolve/SmallDenseLinearAlgebra.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/Sparse.hpp>
#include <resolve/utilities/logger/Logger.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>

namespace ReSolve
{
  using out = io::Logger;

  namespace
  {
    struct RitzBlock
    {
      index_type first{0};
      index_type width{0};
      real_type  magnitude{0.0};
    };

    bool isNegative(index_type value)
    {
      return static_cast<long double>(value) < 0.0L;
    }

    bool fitsLapackInt(index_type value)
    {
      return static_cast<long double>(value)
             <= static_cast<long double>(std::numeric_limits<int>::max());
    }

    bool isComplexEigenvalue(real_type alphar,
                             real_type alphai)
    {
      using namespace constants;
      const real_type scale = std::max(ONE, std::hypot(alphar, alphai));
      return std::abs(alphai) > 100.0 * MACHINE_EPSILON * scale;
    }

    real_type generalizedEigenvalueMagnitude(real_type alphar,
                                             real_type alphai,
                                             real_type beta)
    {
      if (beta == 0.0)
      {
        return std::numeric_limits<real_type>::infinity();
      }
      return std::hypot(alphar, alphai) / std::abs(beta);
    }

    int selectSmallestRitzVectors(index_type              dimension,
                                  index_type              target,
                                  index_type              capacity,
                                  const real_type*        alphar,
                                  const real_type*        alphai,
                                  const real_type*        beta,
                                  const real_type*        eigenvectors,
                                  std::vector<real_type>& selected_vectors,
                                  index_type&             selected_count)
    {
      selected_count = 0;
      selected_vectors.clear();

      std::vector<RitzBlock> blocks;
      for (index_type i = 0; i < dimension;)
      {
        RitzBlock block;
        block.first     = i;
        block.width     = 1;
        block.magnitude = generalizedEigenvalueMagnitude(alphar[i], alphai[i], beta[i]);
        if (isComplexEigenvalue(alphar[i], alphai[i]))
        {
          if (alphai[i] <= 0.0 || i + 1 >= dimension
              || alphai[i + 1] >= 0.0
              || !isComplexEigenvalue(alphar[i + 1], alphai[i + 1]))
          {
            return 1;
          }
          block.width = 2;
        }
        blocks.push_back(block);
        i += block.width;
      }

      std::stable_sort(blocks.begin(),
                       blocks.end(),
                       [](const RitzBlock& left, const RitzBlock& right)
                       {
                         return left.magnitude < right.magnitude;
                       });

      std::vector<index_type> selected;
      selected.reserve(static_cast<std::size_t>(capacity));
      for (const RitzBlock& block : blocks)
      {
        if (selected.size() + static_cast<std::size_t>(block.width)
            > static_cast<std::size_t>(capacity))
        {
          continue;
        }
        for (index_type j = 0; j < block.width; ++j)
        {
          selected.push_back(block.first + j);
        }
        // Retain a complex pair in full even when it crosses the requested
        // target dimension.
        if (selected.size() >= static_cast<std::size_t>(target))
        {
          break;
        }
      }

      selected_count                   = static_cast<index_type>(selected.size());
      const std::size_t dimension_size = static_cast<std::size_t>(dimension);
      selected_vectors.resize(dimension_size * selected.size());
      for (index_type column = 0; column < selected_count; ++column)
      {
        const index_type source = selected[static_cast<std::size_t>(column)];
        std::copy_n(eigenvectors + static_cast<std::size_t>(source) * dimension_size,
                    dimension_size,
                    selected_vectors.data()
                        + static_cast<std::size_t>(column) * dimension_size);
      }
      return 0;
    }
  } // namespace

  LinSolverIterativeGCRODR::LinSolverIterativeGCRODR(MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
    : GS_(gs)
  {
    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeGCRODR::LinSolverIterativeGCRODR(index_type     restart,
                                                     index_type     recycle_dim,
                                                     real_type      tol,
                                                     index_type     maxit,
                                                     index_type     conv_cond,
                                                     MatrixHandler* matrix_handler,
                                                     VectorHandler* vector_handler,
                                                     GramSchmidt*   gs)
    : restart_(restart),
      recycle_dim_(recycle_dim),
      conv_cond_(conv_cond),
      GS_(gs)
  {
    tol_            = tol;
    maxit_          = maxit;
    matrix_handler_ = matrix_handler;
    vector_handler_ = vector_handler;
    setMemorySpace();
    initParamList();
  }

  LinSolverIterativeGCRODR::~LinSolverIterativeGCRODR()
  {
    if (is_solver_set_)
    {
      freeSolverData();
    }
  }

  int LinSolverIterativeGCRODR::setup(matrix::Sparse* A)
  {
    if (A == nullptr || !handlers_compatible_ || GS_ == nullptr
        || !parametersValid())
    {
      out::error() << "Invalid GCRO-DR setup data or parameters.\n";
      return 1;
    }
    if (A->getNumRows() == 0 || A->getNumRows() != A->getNumColumns())
    {
      out::error() << "GCRO-DR requires a square system matrix.\n";
      return 1;
    }

    const bool size_changed = n_ != A->getNumRows();
    if (size_changed && is_solver_set_)
    {
      freeSolverData();
      is_solver_set_ = false;
    }

    A_ = A;
    n_ = A->getNumRows();

    if (!is_solver_set_)
    {
      allocateSolverData();
      is_solver_set_ = true;
      GS_->setup(n_, restart_);
    }
    else if (!GS_->isSetupComplete())
    {
      GS_->setup(n_, restart_);
    }

    if (size_changed)
    {
      active_recycle_dim_ = 0;
    }
    last_cycle_dim_      = 0;
    recycle_image_valid_ = false;
    return 0;
  }

  int LinSolverIterativeGCRODR::solve(vector_type* rhs, vector_type* x)
  {
    using namespace constants;
    RESOLVE_PROFILE_SCOPE("GCRODR.solve");

    total_iters_    = 0;
    last_cycle_dim_ = 0;

    if (!is_solver_set_ || A_ == nullptr || rhs == nullptr || x == nullptr
        || !parametersValid())
    {
      out::error() << "GCRO-DR is not set up or received invalid solve data.\n";
      return 1;
    }
    if (rhs->getSize() != n_ || x->getSize() != n_
        || !rhs->isAllocated(memspace_) || !x->isAllocated(memspace_))
    {
      out::error() << "GCRO-DR vector dimensions or memory spaces are invalid.\n";
      return 1;
    }
    if (preconditioner_ == nullptr)
    {
      out::error() << "Preconditioner not set for GCRO-DR solver.\n";
      return 1;
    }
    if (preconditioner_->getSide() != Preconditioner::Side::RIGHT)
    {
      out::error() << "GCRO-DR currently supports right preconditioning only.\n";
      return 1;
    }
    if (active_recycle_dim_ != 0 && !recycle_image_valid_)
    {
      RESOLVE_PROFILE_SCOPE("GCRODR.recycle.refresh");
      if (refreshRecycleSpace() != 0)
      {
        out::error() << "GCRO-DR failed to refresh the recycle space for the "
                     << "current matrix or preconditioner.\n";
        clearRecycleSpace();
        return 1;
      }
    }

    // For a subsequent right-hand side, retain the existing recycle space
    // and start directly with projected GCRO cycles.
    if (active_recycle_dim_ != 0)
    {
      real_type rhs_norm = vector_handler_->dot(rhs, rhs, memspace_);
      rhs_norm           = std::sqrt(std::max(ZERO, rhs_norm));

      rhs->copyToExternal(vec_work_->getData(memspace_), 0, memspace_, memspace_);
      if (matrix_handler_->matvec(A_, x, vec_work_, &MINUS_ONE, &ONE, memspace_) != 0)
      {
        out::error() << "GCRO-DR failed to compute the recycled solve residual.\n";
        return 1;
      }
      real_type residual_norm = vector_handler_->dot(vec_work_, vec_work_, memspace_);
      residual_norm           = std::sqrt(std::max(ZERO, residual_norm));
      initial_residual_norm_  = relativeResidual(residual_norm, rhs_norm);
      final_residual_norm_    = initial_residual_norm_;

      if (!hasConverged(residual_norm, rhs_norm)
          && runRecycledCycles(rhs, x, rhs_norm, residual_norm) != 0)
      {
        return 1;
      }

      io::Logger::misc() << "GCRO-DR end of recycled solve: true norm of residual "
                         << std::scientific << std::setprecision(16)
                         << residual_norm << "\n";
      return 0;
    }

    const std::size_t restart = static_cast<std::size_t>(restart_);
    std::fill_n(h_H_, restart * (restart + 1), ZERO);
    std::fill_n(h_G_, restart * (restart + 1), ZERO);
    std::fill_n(h_c_, restart, ZERO);
    std::fill_n(h_s_, restart, ZERO);
    std::fill_n(h_rs_, restart + 1, ZERO);
    vec_V_->setToZero(memspace_);
    vec_Z_->setToZero(memspace_);
    vec_work_->setToZero(memspace_);

    vector_type vec_v(n_);
    vector_type vec_w(n_);
    vector_type x_initial(n_);
    x_initial.allocate(memspace_);
    x_initial.copyFromExternal(x, memspace_, memspace_);

    real_type x_norm            = vector_handler_->dot(x, x, memspace_);
    x_norm                      = std::sqrt(std::max(ZERO, x_norm));
    bool preserve_initial_guess = x_norm > MACHINE_EPSILON;

    // Form the true initial residual in the first Arnoldi vector.
    rhs->copyToExternal(vec_V_->getData(0, memspace_), 0, memspace_, memspace_);
    vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
    if (matrix_handler_->matvec(A_, x, &vec_v, &MINUS_ONE, &ONE, memspace_) != 0)
    {
      out::error() << "GCRO-DR failed to compute the initial residual.\n";
      return 1;
    }

    real_type residual_norm = vector_handler_->dot(&vec_v, &vec_v, memspace_);
    residual_norm           = std::sqrt(std::max(ZERO, residual_norm));
    real_type rhs_norm      = vector_handler_->dot(rhs, rhs, memspace_);
    rhs_norm                = std::sqrt(std::max(ZERO, rhs_norm));

    // Match the existing GMRES behavior: discard a nonzero guess if it is
    // worse than the zero vector.
    if (preserve_initial_guess && residual_norm > rhs_norm)
    {
      out::warning() << "Initial guess has a larger residual than the zero vector. "
                     << "Ignoring initial guess.\n";
      x->setToZero(memspace_);
      x_initial.copyFromExternal(x, memspace_, memspace_);
      preserve_initial_guess = false;

      rhs->copyToExternal(vec_V_->getData(0, memspace_), 0, memspace_, memspace_);
      if (matrix_handler_->matvec(A_, x, &vec_v, &MINUS_ONE, &ONE, memspace_) != 0)
      {
        out::error() << "GCRO-DR failed to recompute the initial residual.\n";
        return 1;
      }
      residual_norm = vector_handler_->dot(&vec_v, &vec_v, memspace_);
      residual_norm = std::sqrt(std::max(ZERO, residual_norm));
    }

    const real_type accepted_initial_norm = residual_norm;
    initial_residual_norm_                = relativeResidual(residual_norm, rhs_norm);
    final_residual_norm_                  = initial_residual_norm_;

    io::Logger::misc() << "GCRO-DR it 0: norm of residual "
                       << std::scientific << std::setprecision(16)
                       << residual_norm << " norm of rhs: " << rhs_norm << "\n";

    if (hasConverged(residual_norm, rhs_norm))
    {
      return 0;
    }

    vector_handler_->scal(ONE / residual_norm, &vec_v, memspace_);
    h_rs_[0] = residual_norm;

    bool happy_breakdown = false;
    for (index_type i = 0; i < restart_ && total_iters_ < maxit_; ++i)
    {
      vec_v.setData(vec_V_->getData(i, memspace_), memspace_);

      // Fixed right preconditioning: z_i = M^{-1} v_i and
      // v_{i+1} = A z_i. Only one z vector is needed because M is linear and
      // the final basis combination can be preconditioned once.
      if (preconditioner_->apply(&vec_v, vec_Z_) != 0)
      {
        out::error() << "GCRO-DR preconditioner application failed.\n";
        return 1;
      }
      vec_w.setData(vec_V_->getData(i + 1, memspace_), memspace_);
      if (matrix_handler_->matvec(A_, vec_Z_, &vec_w, &ONE, &ZERO, memspace_) != 0)
      {
        out::error() << "GCRO-DR Arnoldi matrix-vector product failed.\n";
        return 1;
      }

      const int         gs_status     = GS_->orthogonalize(n_, vec_V_, h_H_, i);
      const std::size_t column_offset = static_cast<std::size_t>(i) * (restart + 1);
      const real_type   subdiagonal   = h_H_[column_offset + static_cast<std::size_t>(i + 1)];
      happy_breakdown                 = gs_status != 0 && std::abs(subdiagonal) <= MACHINE_EPSILON;
      if (gs_status != 0 && !happy_breakdown)
      {
        out::error() << "GCRO-DR Arnoldi orthogonalization failed.\n";
        return 1;
      }

      // Keep h_H_ unrotated for the harmonic Ritz extraction in Step 3.
      // h_G_ is the working copy used by the least-squares QR factorization.
      std::copy_n(h_H_ + column_offset,
                  static_cast<std::size_t>(i + 2),
                  h_G_ + column_offset);

      for (index_type k = 0; k < i; ++k)
      {
        const std::size_t row         = static_cast<std::size_t>(k);
        const real_type   t           = h_G_[column_offset + row];
        h_G_[column_offset + row]     = h_c_[k] * t + h_s_[k] * h_G_[column_offset + row + 1];
        h_G_[column_offset + row + 1] = -h_s_[k] * t + h_c_[k] * h_G_[column_offset + row + 1];
      }

      const std::size_t diagonal_row = static_cast<std::size_t>(i);
      const real_type   diagonal     = h_G_[column_offset + diagonal_row];
      const real_type   next         = h_G_[column_offset + diagonal_row + 1];
      const real_type   gamma        = std::hypot(diagonal, next);
      if (gamma == ZERO || !std::isfinite(gamma))
      {
        out::error() << "GCRO-DR encountered a singular Arnoldi least-squares problem.\n";
        return 1;
      }

      h_c_[i]      = diagonal / gamma;
      h_s_[i]      = next / gamma;
      h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
      h_rs_[i] *= h_c_[i];
      h_G_[column_offset + diagonal_row]     = gamma;
      h_G_[column_offset + diagonal_row + 1] = ZERO;

      ++total_iters_;
      last_cycle_dim_ = i + 1;
      residual_norm   = std::abs(h_rs_[i + 1]);

      io::Logger::misc() << "GCRO-DR it " << total_iters_
                         << ": estimated norm of residual "
                         << std::scientific << std::setprecision(16)
                         << residual_norm << "\n";

      if (hasConverged(residual_norm, rhs_norm) || happy_breakdown)
      {
        break;
      }
    }

    if (last_cycle_dim_ == 0)
    {
      out::error() << "GCRO-DR did not perform an Arnoldi iteration.\n";
      return 1;
    }

    const int solve_status = SmallDenseLinearAlgebra::triangularSolve(
        'U',
        'N',
        'N',
        last_cycle_dim_,
        1,
        h_G_,
        restart_ + 1,
        h_rs_,
        restart_ + 1);
    if (solve_status != 0)
    {
      out::error() << "GCRO-DR failed to solve the Arnoldi least-squares problem.\n";
      return 1;
    }

    // x <- x + M^{-1} V_m y. vec_work_ holds V_m y and vec_Z_ holds
    // its preconditioned image.
    vec_work_->setToZero(memspace_);
    for (index_type j = 0; j < last_cycle_dim_; ++j)
    {
      vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
      vector_handler_->axpy(h_rs_[j], &vec_v, vec_work_, memspace_);
    }
    if (preconditioner_->apply(vec_work_, vec_Z_) != 0)
    {
      out::error() << "GCRO-DR final preconditioner application failed.\n";
      return 1;
    }
    vector_handler_->axpy(ONE, vec_Z_, x, memspace_);

    // Compute and report the true residual without overwriting the Arnoldi
    // basis retained in vec_V_.
    rhs->copyToExternal(vec_work_->getData(memspace_), 0, memspace_, memspace_);
    if (matrix_handler_->matvec(A_, x, vec_work_, &MINUS_ONE, &ONE, memspace_) != 0)
    {
      out::error() << "GCRO-DR failed to compute the final residual.\n";
      return 1;
    }
    real_type final_norm = vector_handler_->dot(vec_work_, vec_work_, memspace_);
    final_norm           = std::sqrt(std::max(ZERO, final_norm));

    if (preserve_initial_guess && final_norm > accepted_initial_norm)
    {
      out::warning() << "GCRO-DR did not improve the initial guess. "
                     << "Returning the initial guess.\n";
      x->copyFromExternal(&x_initial, memspace_, memspace_);
      final_norm = accepted_initial_norm;
    }

    final_residual_norm_ = relativeResidual(final_norm, rhs_norm);
    recycle_image_valid_ = false;

    if (buildInitialRecycleSpace() != 0)
    {
      out::error() << "GCRO-DR failed to construct the initial recycle space.\n";
      clearRecycleSpace();
      return 1;
    }

    if (!hasConverged(final_norm, rhs_norm) && total_iters_ < maxit_)
    {
      if (runRecycledCycles(rhs, x, rhs_norm, final_norm) != 0)
      {
        return 1;
      }
    }

    io::Logger::misc() << "GCRO-DR end of solve: true norm of residual "
                       << std::scientific << std::setprecision(16)
                       << final_norm << "\n";
    return 0;
  }

  int LinSolverIterativeGCRODR::resetMatrix(matrix::Sparse* new_A)
  {
    if (new_A == nullptr)
    {
      out::error() << "Cannot reset GCRO-DR with a null matrix.\n";
      return 1;
    }
    if (new_A->getNumRows() == 0 || new_A->getNumRows() != new_A->getNumColumns())
    {
      out::error() << "GCRO-DR requires a square system matrix.\n";
      return 1;
    }
    if (!is_solver_set_ || n_ != new_A->getNumRows())
    {
      return setup(new_A);
    }

    A_                   = new_A;
    last_cycle_dim_      = 0;
    recycle_image_valid_ = false;
    matrix_handler_->setValuesChanged(true, memspace_);
    if (active_recycle_dim_ != 0 && preconditioner_ != nullptr)
    {
      if (preconditioner_->getSide() != Preconditioner::Side::RIGHT
          || refreshRecycleSpace() != 0)
      {
        out::error() << "GCRO-DR failed to refresh after resetting the matrix.\n";
        clearRecycleSpace();
        return 1;
      }
    }
    return 0;
  }

  int LinSolverIterativeGCRODR::setPreconditioner(Preconditioner* preconditioner)
  {
    const int status = LinSolverIterative::setPreconditioner(preconditioner);
    if (status == 0)
    {
      last_cycle_dim_      = 0;
      recycle_image_valid_ = false;
    }
    return status;
  }

  int LinSolverIterativeGCRODR::setOrthogonalization(GramSchmidt* gs)
  {
    if (gs == nullptr)
    {
      return 1;
    }
    GS_             = gs;
    last_cycle_dim_ = 0;
    if (is_solver_set_)
    {
      return GS_->setup(n_, restart_);
    }
    return 0;
  }

  int LinSolverIterativeGCRODR::setRestart(index_type restart)
  {
    if (isNegative(restart) || !fitsLapackInt(restart) || restart == 0
        || recycle_dim_ >= restart)
    {
      out::error() << "GCRO-DR restart must be positive and greater than recycle_dim.\n";
      return 1;
    }
    if (restart_ == restart)
    {
      return 0;
    }

    if (is_solver_set_)
    {
      freeSolverData();
    }
    restart_             = restart;
    active_recycle_dim_  = 0;
    last_cycle_dim_      = 0;
    recycle_image_valid_ = false;
    if (is_solver_set_)
    {
      allocateSolverData();
      GS_->setup(n_, restart_);
    }
    return 0;
  }

  int LinSolverIterativeGCRODR::setRecycleDimension(index_type recycle_dim)
  {
    if (isNegative(recycle_dim) || !fitsLapackInt(recycle_dim)
        || recycle_dim >= restart_)
    {
      out::error() << "GCRO-DR recycle_dim must be smaller than restart.\n";
      return 1;
    }
    if (recycle_dim_ == recycle_dim)
    {
      return 0;
    }

    if (is_solver_set_)
    {
      freeSolverData();
    }
    recycle_dim_         = recycle_dim;
    active_recycle_dim_  = 0;
    last_cycle_dim_      = 0;
    recycle_image_valid_ = false;
    if (is_solver_set_)
    {
      allocateSolverData();
    }
    return 0;
  }

  int LinSolverIterativeGCRODR::setConvergenceCondition(index_type conv_cond)
  {
    if (isNegative(conv_cond) || conv_cond > 2)
    {
      out::error() << "GCRO-DR convergence condition must be 0, 1, or 2.\n";
      return 1;
    }
    conv_cond_ = conv_cond;
    return 0;
  }

  int LinSolverIterativeGCRODR::clearRecycleSpace()
  {
    if (vec_U_ != nullptr)
    {
      vec_U_->setToZero(memspace_);
    }
    if (vec_C_ != nullptr)
    {
      vec_C_->setToZero(memspace_);
    }
    if (vec_Y_ != nullptr)
    {
      vec_Y_->setToZero(memspace_);
    }
    active_recycle_dim_  = 0;
    recycle_image_valid_ = false;
    return 0;
  }

  index_type LinSolverIterativeGCRODR::getRestart() const
  {
    return restart_;
  }

  index_type LinSolverIterativeGCRODR::getRecycleDimension() const
  {
    return recycle_dim_;
  }

  index_type LinSolverIterativeGCRODR::getActiveRecycleDimension() const
  {
    return active_recycle_dim_;
  }

  index_type LinSolverIterativeGCRODR::getConvCond() const
  {
    return conv_cond_;
  }

  const LinSolverIterativeGCRODR::vector_type*
  LinSolverIterativeGCRODR::getRecycleBasis() const
  {
    return vec_U_;
  }

  const LinSolverIterativeGCRODR::vector_type*
  LinSolverIterativeGCRODR::getRecycleImage() const
  {
    return vec_C_;
  }

  int LinSolverIterativeGCRODR::setCliParam(const std::string id, const std::string value)
  {
    switch (getParamId(id))
    {
    case TOL:
    {
      const real_type tolerance = static_cast<real_type>(std::atof(value.c_str()));
      if (tolerance < 0.0)
      {
        return 1;
      }
      setTol(tolerance);
      return 0;
    }
    case MAXIT:
    {
      const long long maxit = std::atoll(value.c_str());
      if (maxit <= 0
          || static_cast<unsigned long long>(maxit)
                 > static_cast<unsigned long long>(std::numeric_limits<index_type>::max()))
      {
        return 1;
      }
      setMaxit(static_cast<index_type>(maxit));
      return 0;
    }
    case RESTART:
      return setRestart(static_cast<index_type>(std::atoi(value.c_str())));
    case CONV_COND:
      return setConvergenceCondition(static_cast<index_type>(std::atoi(value.c_str())));
    case RECYCLE_DIM:
      return setRecycleDimension(static_cast<index_type>(std::atoi(value.c_str())));
    default:
      out::error() << "Trying to set unknown GCRO-DR parameter " << id << "\n";
      return 1;
    }
  }

  std::string LinSolverIterativeGCRODR::getCliParamString(const std::string id) const
  {
    out::error() << "Trying to get unknown GCRO-DR string parameter " << id << "\n";
    return "";
  }

  index_type LinSolverIterativeGCRODR::getCliParamInt(const std::string id) const
  {
    switch (getParamId(id))
    {
    case MAXIT:
      return getMaxit();
    case RESTART:
      return getRestart();
    case CONV_COND:
      return getConvCond();
    case RECYCLE_DIM:
      return getRecycleDimension();
    default:
      out::error() << "Trying to get unknown GCRO-DR integer parameter " << id << "\n";
      return static_cast<index_type>(-1);
    }
  }

  real_type LinSolverIterativeGCRODR::getCliParamReal(const std::string id) const
  {
    if (getParamId(id) == TOL)
    {
      return getTol();
    }
    out::error() << "Trying to get unknown GCRO-DR real parameter " << id << "\n";
    return std::numeric_limits<real_type>::quiet_NaN();
  }

  bool LinSolverIterativeGCRODR::getCliParamBool(const std::string id) const
  {
    out::error() << "Trying to get unknown GCRO-DR boolean parameter " << id << "\n";
    return false;
  }

  int LinSolverIterativeGCRODR::printCliParam(const std::string id) const
  {
    switch (getParamId(id))
    {
    case TOL:
      std::cout << getTol() << "\n";
      break;
    case MAXIT:
      std::cout << getMaxit() << "\n";
      break;
    case RESTART:
      std::cout << getRestart() << "\n";
      break;
    case CONV_COND:
      std::cout << getConvCond() << "\n";
      break;
    case RECYCLE_DIM:
      std::cout << getRecycleDimension() << "\n";
      break;
    default:
      out::error() << "Trying to print unknown GCRO-DR parameter " << id << "\n";
      return 1;
    }
    return 0;
  }

  int LinSolverIterativeGCRODR::allocateSolverData()
  {
    const std::size_t restart          = static_cast<std::size_t>(restart_);
    const index_type  recycle_capacity = recycleCapacity();
    const std::size_t recycle          = static_cast<std::size_t>(recycle_capacity);

    vec_V_    = new vector_type(n_, restart_ + 1);
    vec_Z_    = new vector_type(n_);
    vec_work_ = new vector_type(n_);
    vec_V_->allocate(memspace_);
    vec_Z_->allocate(memspace_);
    vec_work_->allocate(memspace_);

    if (recycle_capacity > 0)
    {
      vec_U_          = new vector_type(n_, recycle_capacity);
      vec_C_          = new vector_type(n_, recycle_capacity);
      vec_Y_          = new vector_type(n_, recycle_capacity);
      vec_B_          = new vector_type(recycle_capacity, restart_);
      vec_projection_ = new vector_type(recycle_capacity);
      vec_U_->allocate(memspace_);
      vec_C_->allocate(memspace_);
      vec_Y_->allocate(memspace_);
      vec_B_->allocate(memspace_);
      vec_projection_->allocate(memspace_);
      if (memspace_ == memory::DEVICE)
      {
        vec_augmented_image_       = new vector_type(n_, restart_ + 1);
        vec_augmented_domain_      = new vector_type(n_, restart_);
        vec_augmented_product_     = new vector_type(restart_ + 1, restart_);
        vec_image_coefficients_    = new vector_type(restart_ + 1, restart_);
        vec_augmented_image_->allocate(memspace_);
        vec_augmented_domain_->allocate(memspace_);
        vec_augmented_product_->allocate(memspace_);
        vec_image_coefficients_->allocate(memspace_);
      }
    }

    h_H_  = new real_type[restart * (restart + 1)]();
    h_B_  = recycle > 0 ? new real_type[recycle * restart]() : nullptr;
    h_G_  = new real_type[restart * (restart + 1)]();
    h_c_  = new real_type[restart]();
    h_s_  = new real_type[restart]();
    h_rs_ = new real_type[restart + 1]();
    return 0;
  }

  int LinSolverIterativeGCRODR::freeSolverData()
  {
    delete vec_U_;
    delete vec_C_;
    delete vec_Y_;
    delete vec_V_;
    delete vec_Z_;
    delete vec_work_;
    delete vec_B_;
    delete vec_projection_;
    delete vec_augmented_image_;
    delete vec_augmented_domain_;
    delete vec_augmented_product_;
    delete vec_image_coefficients_;
    delete[] h_H_;
    delete[] h_B_;
    delete[] h_G_;
    delete[] h_c_;
    delete[] h_s_;
    delete[] h_rs_;

    vec_U_                  = nullptr;
    vec_C_                  = nullptr;
    vec_Y_                  = nullptr;
    vec_V_                  = nullptr;
    vec_Z_                  = nullptr;
    vec_work_               = nullptr;
    vec_B_                  = nullptr;
    vec_projection_         = nullptr;
    vec_augmented_image_    = nullptr;
    vec_augmented_domain_   = nullptr;
    vec_augmented_product_  = nullptr;
    vec_image_coefficients_ = nullptr;
    h_H_                    = nullptr;
    h_B_                    = nullptr;
    h_G_                    = nullptr;
    h_c_                    = nullptr;
    h_s_                    = nullptr;
    h_rs_                   = nullptr;
    return 0;
  }

  int LinSolverIterativeGCRODR::buildInitialRecycleSpace()
  {
    using namespace constants;

    if (last_cycle_dim_ == 0 || recycle_dim_ == 0)
    {
      active_recycle_dim_  = 0;
      recycle_image_valid_ = false;
      return 0;
    }
    if (vec_U_ == nullptr || vec_C_ == nullptr || vec_Y_ == nullptr
        || preconditioner_ == nullptr)
    {
      return 1;
    }

    const index_type  m        = last_cycle_dim_;
    const index_type  target   = std::min(recycle_dim_, m);
    const index_type  capacity = std::min(recycleCapacity(), m);
    const std::size_t m_size   = static_cast<std::size_t>(m);
    const std::size_t h_ld     = static_cast<std::size_t>(restart_ + 1);

    // For the initial Arnoldi relation A M^{-1} V_m = V_{m+1} Hbar_m,
    // harmonic Ritz coefficient vectors satisfy
    //
    //   Hbar_m^T Hbar_m p = theta H_m^T p.
    //
    // Both small matrices are stored in LAPACK column-major form.
    std::vector<real_type> eigen_left(m_size * m_size, ZERO);
    std::vector<real_type> eigen_right(m_size * m_size, ZERO);
    for (index_type column = 0; column < m; ++column)
    {
      for (index_type row = 0; row < m; ++row)
      {
        real_type product = ZERO;
        for (index_type k = 0; k <= m; ++k)
        {
          product += h_H_[static_cast<std::size_t>(row) * h_ld
                          + static_cast<std::size_t>(k)]
                     * h_H_[static_cast<std::size_t>(column) * h_ld
                            + static_cast<std::size_t>(k)];
        }
        eigen_left[static_cast<std::size_t>(column) * m_size
                   + static_cast<std::size_t>(row)] = product;

        eigen_right[static_cast<std::size_t>(column) * m_size
                    + static_cast<std::size_t>(row)] =
            h_H_[static_cast<std::size_t>(row) * h_ld
                 + static_cast<std::size_t>(column)];
      }
    }

    std::vector<real_type> alphar(m_size, ZERO);
    std::vector<real_type> alphai(m_size, ZERO);
    std::vector<real_type> beta(m_size, ZERO);
    std::vector<real_type> eigenvectors(m_size * m_size, ZERO);
    const int              eigen_status = SmallDenseLinearAlgebra::generalizedEigenvectors(
        m,
        eigen_left.data(),
        m,
        eigen_right.data(),
        m,
        alphar.data(),
        alphai.data(),
        beta.data(),
        eigenvectors.data(),
        m);
    if (eigen_status != 0)
    {
      return 1;
    }

    std::vector<real_type> P;
    index_type             recycle = 0;
    if (selectSmallestRitzVectors(m,
                                  target,
                                  capacity,
                                  alphar.data(),
                                  alphai.data(),
                                  beta.data(),
                                  eigenvectors.data(),
                                  P,
                                  recycle)
        != 0)
    {
      return 1;
    }
    if (recycle == 0)
    {
      out::warning() << "No complete real harmonic Ritz block fits recycle_dim.\n";
      clearRecycleSpace();
      return 0;
    }

    const std::size_t recycle_size = static_cast<std::size_t>(recycle);

    // QR factorization of Hbar_m P. The resulting Q produces an orthonormal
    // recycle image C = V_{m+1} Q.
    std::vector<real_type> Q((m_size + 1) * recycle_size, ZERO);
    for (index_type column = 0; column < recycle; ++column)
    {
      for (index_type row = 0; row <= m; ++row)
      {
        real_type value = ZERO;
        for (index_type j = 0; j < m; ++j)
        {
          value += h_H_[static_cast<std::size_t>(j) * h_ld
                        + static_cast<std::size_t>(row)]
                   * P[static_cast<std::size_t>(column) * m_size
                       + static_cast<std::size_t>(j)];
        }
        Q[static_cast<std::size_t>(column) * (m_size + 1)
          + static_cast<std::size_t>(row)] = value;
      }
    }

    std::vector<real_type> R(recycle_size * recycle_size, ZERO);
    if (SmallDenseLinearAlgebra::thinQr(m + 1,
                                        recycle,
                                        Q.data(),
                                        m + 1,
                                        R.data(),
                                        recycle)
        != 0)
    {
      return 1;
    }

    // Compute (P R^{-1})^T by solving R^T X^T = P^T. These coefficients
    // build the transformed recycle basis Y=V_m P R^{-1}, followed by
    // U=M^{-1}Y.
    std::vector<real_type> coefficients(recycle_size * m_size, ZERO);
    for (index_type column = 0; column < m; ++column)
    {
      for (index_type row = 0; row < recycle; ++row)
      {
        coefficients[static_cast<std::size_t>(column) * recycle_size
                     + static_cast<std::size_t>(row)] =
            P[static_cast<std::size_t>(row) * m_size
              + static_cast<std::size_t>(column)];
      }
    }
    if (SmallDenseLinearAlgebra::triangularSolve('U',
                                                 'T',
                                                 'N',
                                                 recycle,
                                                 m,
                                                 R.data(),
                                                 recycle,
                                                 coefficients.data(),
                                                 recycle)
        != 0)
    {
      return 1;
    }

    vec_U_->setToZero(memspace_);
    vec_C_->setToZero(memspace_);
    vec_Y_->setToZero(memspace_);
    vector_type vec_v(n_);
    vector_type vec_u(n_);
    vector_type vec_c(n_);
    vector_type vec_y(n_);
    for (index_type column = 0; column < recycle; ++column)
    {
      vec_c.setData(vec_C_->getData(column, memspace_), memspace_);
      for (index_type j = 0; j <= m; ++j)
      {
        vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
        vector_handler_->axpy(
            Q[static_cast<std::size_t>(column) * (m_size + 1)
              + static_cast<std::size_t>(j)],
            &vec_v,
            &vec_c,
            memspace_);
      }

      vec_y.setData(vec_Y_->getData(column, memspace_), memspace_);
      for (index_type j = 0; j < m; ++j)
      {
        vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
        vector_handler_->axpy(
            coefficients[static_cast<std::size_t>(j) * recycle_size
                         + static_cast<std::size_t>(column)],
            &vec_v,
            &vec_y,
            memspace_);
      }
      vec_u.setData(vec_U_->getData(column, memspace_), memspace_);
      if (preconditioner_->apply(&vec_y, &vec_u) != 0)
      {
        return 1;
      }
    }

    active_recycle_dim_  = recycle;
    recycle_image_valid_ = true;
    return 0;
  }

  int LinSolverIterativeGCRODR::updateRecycleSpace()
  {
    using namespace constants;
    RESOLVE_PROFILE_SCOPE("GCRODR.recycle.update");

    if (recycle_dim_ == 0 || last_cycle_dim_ == 0)
    {
      return 0;
    }
    if (active_recycle_dim_ == 0)
    {
      return buildInitialRecycleSpace();
    }
    if (vec_U_ == nullptr || vec_C_ == nullptr || vec_Y_ == nullptr
        || preconditioner_ == nullptr)
    {
      return 1;
    }

    const index_type  old_recycle    = active_recycle_dim_;
    const index_type  krylov         = last_cycle_dim_;
    const index_type  augmented      = old_recycle + krylov;
    const index_type  rows           = augmented + 1;
    const index_type  target         = std::min(recycle_dim_, augmented);
    const index_type  capacity       = std::min(recycleCapacity(), augmented);
    const std::size_t rows_size      = static_cast<std::size_t>(rows);
    const std::size_t augmented_size = static_cast<std::size_t>(augmented);
    const std::size_t h_ld           = static_cast<std::size_t>(restart_ + 1);
    const std::size_t b_ld           = static_cast<std::size_t>(recycleCapacity());

    if (augmented > restart_ || capacity == 0)
    {
      return 1;
    }

    // Apply GCRO-DR to the fixed right-preconditioned operator T=A M^{-1}.
    // Scale the retained operator-domain vectors to unit norm. With
    // X=[Y D,V], U=M^{-1}Y, and W=[C,V_next], the augmented relation is
    //
    //   T X = W G,  G = [D B; 0 Hbar].
    std::vector<real_type> scaling(static_cast<std::size_t>(old_recycle), ZERO);
    vector_type            vec_basis(n_);
    vector_type            vec_w_basis(n_);
    vector_type            vec_v(n_);
    {
      RESOLVE_PROFILE_SCOPE("GCRODR.recycle.scaling_norm_syncs");
      if (memspace_ == memory::DEVICE)
      {
        if (vec_augmented_product_ == nullptr)
        {
          return 1;
        }

        vector_type gram(old_recycle, old_recycle);
        gram.setData(vec_augmented_product_->getData(memspace_), memspace_);
        vector_handler_->gemm('T',
                              n_,
                              vec_Y_,
                              old_recycle,
                              vec_Y_,
                              old_recycle,
                              &gram,
                              memspace_);

        std::vector<real_type> gram_values(
            static_cast<std::size_t>(old_recycle)
                * static_cast<std::size_t>(old_recycle),
            ZERO);
        if (gram.copyToExternal(gram_values.data(), memspace_, memory::HOST)
            != 0)
        {
          return 1;
        }
        for (index_type j = 0; j < old_recycle; ++j)
        {
          const std::size_t diagonal =
              static_cast<std::size_t>(j)
              * static_cast<std::size_t>(old_recycle + 1);
          const real_type norm =
              std::sqrt(std::max(ZERO, gram_values[diagonal]));
          if (norm <= MACHINE_EPSILON || !std::isfinite(norm))
          {
            return 1;
          }
          scaling[static_cast<std::size_t>(j)] = ONE / norm;
        }
      }
      else
      {
        for (index_type j = 0; j < old_recycle; ++j)
        {
          vec_basis.setData(vec_Y_->getData(j, memspace_), memspace_);
          real_type norm = vector_handler_->dot(&vec_basis,
                                                &vec_basis,
                                                memspace_);
          norm = std::sqrt(std::max(ZERO, norm));
          if (norm <= MACHINE_EPSILON || !std::isfinite(norm))
          {
            return 1;
          }
          scaling[static_cast<std::size_t>(j)] = ONE / norm;
        }
      }
    }

    std::vector<real_type> G(rows_size * augmented_size, ZERO);
    for (index_type j = 0; j < old_recycle; ++j)
    {
      G[static_cast<std::size_t>(j) * rows_size
        + static_cast<std::size_t>(j)] = scaling[static_cast<std::size_t>(j)];
    }
    for (index_type column = 0; column < krylov; ++column)
    {
      const std::size_t augmented_column = static_cast<std::size_t>(old_recycle + column);
      for (index_type row = 0; row < old_recycle; ++row)
      {
        G[augmented_column * rows_size + static_cast<std::size_t>(row)] =
            h_B_[static_cast<std::size_t>(column) * b_ld
                 + static_cast<std::size_t>(row)];
      }
      for (index_type row = 0; row <= krylov; ++row)
      {
        G[augmented_column * rows_size
          + static_cast<std::size_t>(old_recycle + row)] =
            h_H_[static_cast<std::size_t>(column) * h_ld
                 + static_cast<std::size_t>(row)];
      }
    }

    // Form F=W^T X in the transformed, right-preconditioned system.
    std::vector<real_type> F(rows_size * augmented_size, ZERO);
    if (memspace_ == memory::DEVICE)
    {
      RESOLVE_PROFILE_SCOPE("GCRODR.recycle.augmented_product_sync");
      if (vec_augmented_image_ == nullptr
          || vec_augmented_domain_ == nullptr
          || vec_augmented_product_ == nullptr)
      {
        return 1;
      }

      // Pack W=[C,V_{m+1}] and X=[Y,V_m] using four contiguous device copies.
      // The first old_recycle columns of X are scaled after the small product
      // is copied to the host, leaving the retained Y basis unchanged.
      vector_type image_recycle(n_, old_recycle);
      vector_type image_krylov(n_, krylov + 1);
      vector_type domain_recycle(n_, old_recycle);
      vector_type domain_krylov(n_, krylov);
      image_recycle.setData(vec_augmented_image_->getData(memspace_), memspace_);
      image_krylov.setData(
          vec_augmented_image_->getData(old_recycle, memspace_), memspace_);
      domain_recycle.setData(vec_augmented_domain_->getData(memspace_), memspace_);
      domain_krylov.setData(
          vec_augmented_domain_->getData(old_recycle, memspace_), memspace_);
      if (image_recycle.copyFromExternal(vec_C_, memspace_, memspace_) != 0
          || image_krylov.copyFromExternal(vec_V_, memspace_, memspace_) != 0
          || domain_recycle.copyFromExternal(vec_Y_, memspace_, memspace_) != 0
          || domain_krylov.copyFromExternal(vec_V_, memspace_, memspace_) != 0)
      {
        return 1;
      }
      vec_augmented_image_->setDataUpdated(memspace_);
      vec_augmented_domain_->setDataUpdated(memspace_);

      vec_augmented_product_->resize(rows);
      vector_handler_->gemm('T',
                            n_,
                            vec_augmented_image_,
                            rows,
                            vec_augmented_domain_,
                            augmented,
                            vec_augmented_product_,
                            memspace_);

      std::vector<real_type> packed_product(
          rows_size * static_cast<std::size_t>(restart_), ZERO);
      if (vec_augmented_product_->copyToExternal(packed_product.data(),
                                                 memspace_,
                                                 memory::HOST)
          != 0)
      {
        return 1;
      }
      for (index_type column = 0; column < augmented; ++column)
      {
        const real_type column_scale =
            column < old_recycle ? scaling[static_cast<std::size_t>(column)]
                                 : ONE;
        const std::size_t column_offset =
            static_cast<std::size_t>(column) * rows_size;
        for (index_type row = 0; row < rows; ++row)
        {
          F[column_offset + static_cast<std::size_t>(row)] =
              column_scale
              * packed_product[column_offset + static_cast<std::size_t>(row)];
        }
      }
    }
    else
    {
      for (index_type column = 0; column < augmented; ++column)
      {
        real_type    column_scale = ONE;
        vector_type* x_column     = nullptr;
        if (column < old_recycle)
        {
          vec_basis.setData(vec_Y_->getData(column, memspace_), memspace_);
          column_scale = scaling[static_cast<std::size_t>(column)];
          x_column     = &vec_basis;
        }
        else
        {
          const index_type krylov_column = column - old_recycle;
          vec_v.setData(vec_V_->getData(krylov_column, memspace_), memspace_);
          x_column = &vec_v;
        }

        const std::size_t column_offset =
            static_cast<std::size_t>(column) * rows_size;
        for (index_type row = 0; row < rows; ++row)
        {
          if (row < old_recycle)
          {
            vec_w_basis.setData(vec_C_->getData(row, memspace_), memspace_);
          }
          else
          {
            vec_w_basis.setData(vec_V_->getData(row - old_recycle, memspace_),
                                memspace_);
          }
          F[column_offset + static_cast<std::size_t>(row)] =
              column_scale
              * vector_handler_->dot(&vec_w_basis, x_column, memspace_);
        }
      }
    }

    // Harmonic Ritz coefficient vectors for the augmented space satisfy
    //
    //   G^T G p = theta G^T W^T X p.
    std::vector<real_type> eigen_left(augmented_size * augmented_size, ZERO);
    std::vector<real_type> eigen_right(augmented_size * augmented_size, ZERO);
    for (index_type column = 0; column < augmented; ++column)
    {
      for (index_type row = 0; row < augmented; ++row)
      {
        real_type left_value  = ZERO;
        real_type right_value = ZERO;
        for (index_type i = 0; i < rows; ++i)
        {
          left_value += G[static_cast<std::size_t>(row) * rows_size
                          + static_cast<std::size_t>(i)]
                        * G[static_cast<std::size_t>(column) * rows_size
                            + static_cast<std::size_t>(i)];
          right_value += G[static_cast<std::size_t>(row) * rows_size
                           + static_cast<std::size_t>(i)]
                         * F[static_cast<std::size_t>(column) * rows_size
                             + static_cast<std::size_t>(i)];
        }
        eigen_left[static_cast<std::size_t>(column) * augmented_size
                   + static_cast<std::size_t>(row)]  = left_value;
        eigen_right[static_cast<std::size_t>(column) * augmented_size
                    + static_cast<std::size_t>(row)] = right_value;
      }
    }

    std::vector<real_type> alphar(augmented_size, ZERO);
    std::vector<real_type> alphai(augmented_size, ZERO);
    std::vector<real_type> beta(augmented_size, ZERO);
    std::vector<real_type> eigenvectors(augmented_size * augmented_size, ZERO);
    if (SmallDenseLinearAlgebra::generalizedEigenvectors(augmented,
                                                         eigen_left.data(),
                                                         augmented,
                                                         eigen_right.data(),
                                                         augmented,
                                                         alphar.data(),
                                                         alphai.data(),
                                                         beta.data(),
                                                         eigenvectors.data(),
                                                         augmented)
        != 0)
    {
      return 1;
    }

    std::vector<real_type> P;
    index_type             recycle = 0;
    if (selectSmallestRitzVectors(augmented,
                                  target,
                                  capacity,
                                  alphar.data(),
                                  alphai.data(),
                                  beta.data(),
                                  eigenvectors.data(),
                                  P,
                                  recycle)
        != 0)
    {
      return 1;
    }
    if (recycle == 0)
    {
      out::warning() << "No complete real harmonic Ritz block fits recycle_dim.\n";
      return 0;
    }

    const std::size_t      recycle_size = static_cast<std::size_t>(recycle);
    std::vector<real_type> Q(rows_size * recycle_size, ZERO);
    for (index_type column = 0; column < recycle; ++column)
    {
      for (index_type row = 0; row < rows; ++row)
      {
        real_type value = ZERO;
        for (index_type j = 0; j < augmented; ++j)
        {
          value += G[static_cast<std::size_t>(j) * rows_size
                     + static_cast<std::size_t>(row)]
                   * P[static_cast<std::size_t>(column) * augmented_size
                       + static_cast<std::size_t>(j)];
        }
        Q[static_cast<std::size_t>(column) * rows_size
          + static_cast<std::size_t>(row)] = value;
      }
    }

    std::vector<real_type> R(recycle_size * recycle_size, ZERO);
    if (SmallDenseLinearAlgebra::thinQr(rows,
                                        recycle,
                                        Q.data(),
                                        rows,
                                        R.data(),
                                        recycle)
        != 0)
    {
      return 1;
    }

    // coefficients=(P R^{-1})^T, stored as recycle-by-augmented.
    std::vector<real_type> coefficients(recycle_size * augmented_size, ZERO);
    for (index_type column = 0; column < augmented; ++column)
    {
      for (index_type row = 0; row < recycle; ++row)
      {
        coefficients[static_cast<std::size_t>(column) * recycle_size
                     + static_cast<std::size_t>(row)] =
            P[static_cast<std::size_t>(row) * augmented_size
              + static_cast<std::size_t>(column)];
      }
    }
    if (SmallDenseLinearAlgebra::triangularSolve('U',
                                                 'T',
                                                 'N',
                                                 recycle,
                                                 augmented,
                                                 R.data(),
                                                 recycle,
                                                 coefficients.data(),
                                                 recycle)
        != 0)
    {
      return 1;
    }

    // Build Y_new=X P R^{-1} in U's storage. The current cycle has already
    // applied its correction, so the old physical U is no longer needed.
    {
      RESOLVE_PROFILE_SCOPE("GCRODR.recycle.build_bases");
      if (memspace_ == memory::DEVICE)
      {
        if (vec_augmented_domain_ == nullptr
            || vec_augmented_image_ == nullptr
            || vec_augmented_product_ == nullptr
            || vec_image_coefficients_ == nullptr)
        {
          return 1;
        }

        // X=[Y D,V_m] is already packed in vec_augmented_domain_. Store the
        // column-major augmented-by-recycle coefficient matrix for
        // Y_new=X P R^{-1} in the product workspace.
        std::vector<real_type> basis_coefficients(
            augmented_size * recycle_size, ZERO);
        for (index_type column = 0; column < recycle; ++column)
        {
          const std::size_t column_offset =
              static_cast<std::size_t>(column) * augmented_size;
          for (index_type j = 0; j < augmented; ++j)
          {
            const real_type source_scale =
                j < old_recycle ? scaling[static_cast<std::size_t>(j)]
                                : ONE;
            basis_coefficients[column_offset + static_cast<std::size_t>(j)] =
                source_scale
                * coefficients[static_cast<std::size_t>(j) * recycle_size
                               + static_cast<std::size_t>(column)];
          }
        }

        vector_type basis_coefficient_matrix(augmented, recycle);
        basis_coefficient_matrix.setData(
            vec_augmented_product_->getData(memspace_), memspace_);
        vector_type image_coefficient_matrix(rows, recycle);
        image_coefficient_matrix.setData(
            vec_image_coefficients_->getData(memspace_), memspace_);
        if (basis_coefficient_matrix.copyFromExternal(basis_coefficients.data(),
                                                      memory::HOST,
                                                      memspace_)
                != 0
            || image_coefficient_matrix.copyFromExternal(Q.data(),
                                                         memory::HOST,
                                                         memspace_)
                   != 0)
        {
          return 1;
        }

        vector_handler_->gemm('N',
                              n_,
                              vec_augmented_domain_,
                              augmented,
                              &basis_coefficient_matrix,
                              recycle,
                              vec_U_,
                              memspace_);
        vector_handler_->gemm('N',
                              n_,
                              vec_augmented_image_,
                              rows,
                              &image_coefficient_matrix,
                              recycle,
                              vec_Y_,
                              memspace_);
      }
      else
      {
        vec_U_->setToZero(memspace_);
        vector_type vec_new(n_);
        vector_type vec_old(n_);
        for (index_type column = 0; column < recycle; ++column)
        {
          vec_new.setData(vec_U_->getData(column, memspace_), memspace_);
          for (index_type j = 0; j < old_recycle; ++j)
          {
            vec_old.setData(vec_Y_->getData(j, memspace_), memspace_);
            const real_type coefficient =
                scaling[static_cast<std::size_t>(j)]
                * coefficients[static_cast<std::size_t>(j) * recycle_size
                               + static_cast<std::size_t>(column)];
            vector_handler_->axpy(coefficient, &vec_old, &vec_new, memspace_);
          }

          for (index_type j = 0; j < krylov; ++j)
          {
            vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
            const real_type coefficient =
                coefficients[static_cast<std::size_t>(old_recycle + j)
                                 * recycle_size
                             + static_cast<std::size_t>(column)];
            vector_handler_->axpy(coefficient, &vec_v, &vec_new, memspace_);
          }
        }

        // The old Y is no longer needed. Use its storage for C_new=W Q.
        vec_Y_->setToZero(memspace_);
        for (index_type column = 0; column < recycle; ++column)
        {
          vec_new.setData(vec_Y_->getData(column, memspace_), memspace_);
          for (index_type j = 0; j < old_recycle; ++j)
          {
            vec_old.setData(vec_C_->getData(j, memspace_), memspace_);
            vector_handler_->axpy(Q[static_cast<std::size_t>(column) * rows_size
                                    + static_cast<std::size_t>(j)],
                                  &vec_old,
                                  &vec_new,
                                  memspace_);
          }
          for (index_type j = 0; j <= krylov; ++j)
          {
            vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
            vector_handler_->axpy(Q[static_cast<std::size_t>(column) * rows_size
                                    + static_cast<std::size_t>(old_recycle + j)],
                                  &vec_v,
                                  &vec_new,
                                  memspace_);
          }
        }
      }
    }

    // Preserve C_new and Y_new, then recover the physical recycle basis
    // U_new=M^{-1}Y_new one column at a time.
    if (vec_C_->copyFromExternal(vec_Y_, memspace_, memspace_) != 0
        || vec_Y_->copyFromExternal(vec_U_, memspace_, memspace_) != 0)
    {
      return 1;
    }
    vec_U_->setToZero(memspace_);
    vector_type vec_y(n_);
    vector_type vec_u(n_);
    {
      RESOLVE_PROFILE_SCOPE("GCRODR.recycle.precondition_bases");
      for (index_type column = 0; column < recycle; ++column)
      {
        vec_y.setData(vec_Y_->getData(column, memspace_), memspace_);
        vec_u.setData(vec_U_->getData(column, memspace_), memspace_);
        if (preconditioner_->apply(&vec_y, &vec_u) != 0)
        {
          return 1;
        }
      }
    }

    active_recycle_dim_  = recycle;
    recycle_image_valid_ = true;
    return 0;
  }

  int LinSolverIterativeGCRODR::refreshRecycleSpace()
  {
    using namespace constants;
    RESOLVE_PROFILE_SCOPE("GCRODR.recycle.refresh_impl");

    if (active_recycle_dim_ == 0)
    {
      recycle_image_valid_ = false;
      return 0;
    }
    if (A_ == nullptr || preconditioner_ == nullptr
        || preconditioner_->getSide() != Preconditioner::Side::RIGHT
        || vec_U_ == nullptr || vec_C_ == nullptr || vec_Y_ == nullptr
        || vec_V_ == nullptr)
    {
      return 1;
    }

    const index_type       recycle      = active_recycle_dim_;
    const std::size_t      recycle_size = static_cast<std::size_t>(recycle);
    std::vector<real_type> R(recycle_size * recycle_size, ZERO);
    vector_type            vec_y(n_);
    vector_type            vec_u(n_);
    vector_type            vec_c(n_);
    vector_type            vec_previous_c(n_);

    // Form the image of the retained transformed basis under the new
    // right-preconditioned operator T=A M^{-1}.
    vec_U_->setToZero(memspace_);
    vec_C_->setToZero(memspace_);
    for (index_type column = 0; column < recycle; ++column)
    {
      vec_y.setData(vec_Y_->getData(column, memspace_), memspace_);
      vec_u.setData(vec_U_->getData(column, memspace_), memspace_);
      vec_c.setData(vec_C_->getData(column, memspace_), memspace_);
      if (preconditioner_->apply(&vec_y, &vec_u) != 0
          || matrix_handler_->matvec(A_,
                                     &vec_u,
                                     &vec_c,
                                     &ONE,
                                     &ZERO,
                                     memspace_)
                 != 0)
      {
        return 1;
      }
    }

    // Twice-applied modified Gram-Schmidt gives C_raw=Q R while keeping the
    // large vectors in their backend memory space.
    for (index_type column = 0; column < recycle; ++column)
    {
      vec_c.setData(vec_C_->getData(column, memspace_), memspace_);
      for (index_type pass = 0; pass < 2; ++pass)
      {
        for (index_type row = 0; row < column; ++row)
        {
          vec_previous_c.setData(vec_C_->getData(row, memspace_), memspace_);
          const real_type coefficient =
              vector_handler_->dot(&vec_previous_c, &vec_c, memspace_);
          R[static_cast<std::size_t>(column) * recycle_size
            + static_cast<std::size_t>(row)] += coefficient;
          vector_handler_->axpy(-coefficient,
                                &vec_previous_c,
                                &vec_c,
                                memspace_);
        }
      }

      real_type norm = vector_handler_->dot(&vec_c, &vec_c, memspace_);
      norm           = std::sqrt(std::max(ZERO, norm));
      if (norm <= MACHINE_EPSILON || !std::isfinite(norm))
      {
        return 1;
      }
      R[static_cast<std::size_t>(column) * recycle_size
        + static_cast<std::size_t>(column)] = norm;
      vector_handler_->scal(ONE / norm, &vec_c, memspace_);
    }

    std::vector<real_type> R_inverse(recycle_size * recycle_size, ZERO);
    for (index_type i = 0; i < recycle; ++i)
    {
      R_inverse[static_cast<std::size_t>(i) * recycle_size
                + static_cast<std::size_t>(i)] = ONE;
    }
    if (SmallDenseLinearAlgebra::triangularSolve('U',
                                                 'N',
                                                 'N',
                                                 recycle,
                                                 recycle,
                                                 R.data(),
                                                 recycle,
                                                 R_inverse.data(),
                                                 recycle)
        != 0)
    {
      return 1;
    }

    // Y_new=Y_old R^{-1}. Use the idle Arnoldi storage as a temporary
    // multivector, then recover U_new=M^{-1}Y_new.
    vec_V_->setToZero(memspace_);
    vector_type vec_new_y(n_);
    for (index_type column = 0; column < recycle; ++column)
    {
      vec_new_y.setData(vec_V_->getData(column, memspace_), memspace_);
      for (index_type source = 0; source < recycle; ++source)
      {
        vec_y.setData(vec_Y_->getData(source, memspace_), memspace_);
        vector_handler_->axpy(
            R_inverse[static_cast<std::size_t>(column) * recycle_size
                      + static_cast<std::size_t>(source)],
            &vec_y,
            &vec_new_y,
            memspace_);
      }
    }
    if (vec_Y_->copyFromExternal(vec_V_, memspace_, memspace_) != 0)
    {
      return 1;
    }

    vec_U_->setToZero(memspace_);
    for (index_type column = 0; column < recycle; ++column)
    {
      vec_y.setData(vec_Y_->getData(column, memspace_), memspace_);
      vec_u.setData(vec_U_->getData(column, memspace_), memspace_);
      if (preconditioner_->apply(&vec_y, &vec_u) != 0)
      {
        return 1;
      }
    }

    recycle_image_valid_ = true;
    return 0;
  }

  int LinSolverIterativeGCRODR::runRecycledCycles(vector_type* rhs,
                                                  vector_type* x,
                                                  real_type    rhs_norm,
                                                  real_type&   final_norm)
  {
    using namespace constants;
    RESOLVE_PROFILE_SCOPE("GCRODR.recycled_cycles");

    const std::size_t restart          = static_cast<std::size_t>(restart_);
    const index_type  recycle_capacity = recycleCapacity();
    vector_type       vec_v(n_);
    vector_type       vec_w(n_);
    vector_type       vec_c(n_);
    vector_type       vec_u(n_);
    std::vector<real_type> correction_coefficients(
        static_cast<std::size_t>(recycle_capacity), ZERO);

    if (memspace_ == memory::DEVICE
        && (vec_B_ == nullptr || vec_projection_ == nullptr))
    {
      return 1;
    }

    while (!hasConverged(final_norm, rhs_norm) && total_iters_ < maxit_)
    {
      const index_type cycle_limit = restart_ - active_recycle_dim_;
      vector_type      vec_b(active_recycle_dim_);
      if (cycle_limit == 0)
      {
        out::error() << "GCRO-DR recycle space leaves no room for Arnoldi vectors.\n";
        return 1;
      }

      std::fill_n(h_H_, restart * (restart + 1), ZERO);
      std::fill_n(h_G_, restart * (restart + 1), ZERO);
      std::fill_n(h_c_, restart, ZERO);
      std::fill_n(h_s_, restart, ZERO);
      std::fill_n(h_rs_, restart + 1, ZERO);
      if (h_B_ != nullptr)
      {
        std::fill_n(h_B_,
                    static_cast<std::size_t>(recycle_capacity) * restart,
                    ZERO);
      }
      vec_V_->setToZero(memspace_);
      vec_Z_->setToZero(memspace_);
      vec_work_->setToZero(memspace_);
      if (memspace_ == memory::DEVICE)
      {
        vec_B_->setToZero(memspace_);
        vec_projection_->resize(active_recycle_dim_);
      }
      last_cycle_dim_ = 0;

      // Compute the true residual and its optimal correction over range(U).
      {
        RESOLVE_PROFILE_SCOPE("GCRODR.cycle.residual_matvec");
        rhs->copyToExternal(vec_V_->getData(0, memspace_), 0, memspace_, memspace_);
        vec_v.setData(vec_V_->getData(0, memspace_), memspace_);
        if (matrix_handler_->matvec(A_, x, &vec_v, &MINUS_ONE, &ONE, memspace_) != 0)
        {
          out::error() << "GCRO-DR failed to compute a recycled-cycle residual.\n";
          return 1;
        }
      }

      if (memspace_ == memory::DEVICE)
      {
        RESOLVE_PROFILE_SCOPE("GCRODR.C_projection.residual");
        // Project the residual over range(C) in two batched passes. Keeping
        // the coefficients in backend memory avoids one host synchronization
        // per recycle vector. The second pass preserves the stability of the
        // previous sequential modified Gram-Schmidt projection.
        vec_b.setData(vec_B_->getData(0, memspace_), memspace_);
        vector_handler_->gemv('T',
                              active_recycle_dim_,
                              ONE,
                              ZERO,
                              vec_C_,
                              &vec_v,
                              &vec_b,
                              memspace_);
        vector_handler_->gemv('N',
                              active_recycle_dim_,
                              MINUS_ONE,
                              ONE,
                              vec_C_,
                              &vec_b,
                              &vec_v,
                              memspace_);
        vector_handler_->gemv('T',
                              active_recycle_dim_,
                              ONE,
                              ZERO,
                              vec_C_,
                              &vec_v,
                              vec_projection_,
                              memspace_);
        vector_handler_->gemv('N',
                              active_recycle_dim_,
                              MINUS_ONE,
                              ONE,
                              vec_C_,
                              vec_projection_,
                              &vec_v,
                              memspace_);
        vector_handler_->axpy(ONE, vec_projection_, &vec_b, memspace_);
        vector_handler_->gemv('N',
                              active_recycle_dim_,
                              ONE,
                              ONE,
                              vec_U_,
                              &vec_b,
                              x,
                              memspace_);
      }
      else
      {
        // Scalar host operations preserve contiguous accesses and are faster
        // than the current CPU GEMV implementation for this small block.
        for (index_type j = 0; j < active_recycle_dim_; ++j)
        {
          vec_c.setData(vec_C_->getData(j, memspace_), memspace_);
          vec_u.setData(vec_U_->getData(j, memspace_), memspace_);
          const real_type coefficient =
              vector_handler_->dot(&vec_c, &vec_v, memspace_);
          vector_handler_->axpy(coefficient, &vec_u, x, memspace_);
          vector_handler_->axpy(-coefficient, &vec_c, &vec_v, memspace_);
        }
      }

      real_type residual_norm = ZERO;
      {
        RESOLVE_PROFILE_SCOPE("GCRODR.cycle.residual_norm_sync");
        residual_norm = vector_handler_->dot(&vec_v, &vec_v, memspace_);
        residual_norm = std::sqrt(std::max(ZERO, residual_norm));
      }
      final_norm              = residual_norm;
      final_residual_norm_    = relativeResidual(final_norm, rhs_norm);

      if (hasConverged(residual_norm, rhs_norm))
      {
        break;
      }

      vector_handler_->scal(ONE / residual_norm, &vec_v, memspace_);
      h_rs_[0] = residual_norm;

      bool happy_breakdown = false;
      for (index_type i = 0; i < cycle_limit && total_iters_ < maxit_; ++i)
      {
        vec_v.setData(vec_V_->getData(i, memspace_), memspace_);
        {
          RESOLVE_PROFILE_SCOPE("GCRODR.arnoldi.preconditioner");
          if (preconditioner_->apply(&vec_v, vec_Z_) != 0)
          {
            out::error() << "GCRO-DR recycled-cycle preconditioner application failed.\n";
            return 1;
          }
        }

        vec_w.setData(vec_V_->getData(i + 1, memspace_), memspace_);
        {
          RESOLVE_PROFILE_SCOPE("GCRODR.arnoldi.spmv");
          if (matrix_handler_->matvec(A_, vec_Z_, &vec_w, &ONE, &ZERO, memspace_) != 0)
          {
            out::error() << "GCRO-DR recycled Arnoldi matrix-vector product failed.\n";
            return 1;
          }
        }

        if (memspace_ == memory::DEVICE)
        {
          RESOLVE_PROFILE_SCOPE("GCRODR.C_projection.arnoldi");
          // Project the Arnoldi image away from C in two batched passes and
          // retain B=C^T A M^{-1}V in backend memory. Each pass uses one GEMV
          // for all coefficients and one GEMV for the basis correction.
          vec_b.setData(vec_B_->getData(i, memspace_), memspace_);
          vector_handler_->gemv('T',
                                active_recycle_dim_,
                                ONE,
                                ZERO,
                                vec_C_,
                                &vec_w,
                                &vec_b,
                                memspace_);
          vector_handler_->gemv('N',
                                active_recycle_dim_,
                                MINUS_ONE,
                                ONE,
                                vec_C_,
                                &vec_b,
                                &vec_w,
                                memspace_);
          vector_handler_->gemv('T',
                                active_recycle_dim_,
                                ONE,
                                ZERO,
                                vec_C_,
                                &vec_w,
                                vec_projection_,
                                memspace_);
          vector_handler_->gemv('N',
                                active_recycle_dim_,
                                MINUS_ONE,
                                ONE,
                                vec_C_,
                                vec_projection_,
                                &vec_w,
                                memspace_);
          vector_handler_->axpy(ONE, vec_projection_, &vec_b, memspace_);
          vec_B_->setDataUpdated(i, memspace_);
        }
        else
        {
          for (index_type j = 0; j < active_recycle_dim_; ++j)
          {
            vec_c.setData(vec_C_->getData(j, memspace_), memspace_);
            const real_type coefficient =
                vector_handler_->dot(&vec_c, &vec_w, memspace_);
            h_B_[static_cast<std::size_t>(i)
                     * static_cast<std::size_t>(recycle_capacity)
                 + static_cast<std::size_t>(j)] = coefficient;
            vector_handler_->axpy(-coefficient, &vec_c, &vec_w, memspace_);
          }
        }

        int gs_status = 0;
        {
          RESOLVE_PROFILE_SCOPE("GCRODR.arnoldi.GramSchmidt");
          gs_status = GS_->orthogonalize(n_, vec_V_, h_H_, i);
        }
        const std::size_t column_offset = static_cast<std::size_t>(i) * (restart + 1);
        const real_type   subdiagonal   = h_H_[column_offset + static_cast<std::size_t>(i + 1)];
        happy_breakdown                 = gs_status != 0 && std::abs(subdiagonal) <= MACHINE_EPSILON;
        if (gs_status != 0 && !happy_breakdown)
        {
          out::error() << "GCRO-DR recycled Arnoldi orthogonalization failed.\n";
          return 1;
        }

        std::copy_n(h_H_ + column_offset,
                    static_cast<std::size_t>(i + 2),
                    h_G_ + column_offset);
        for (index_type k = 0; k < i; ++k)
        {
          const std::size_t row         = static_cast<std::size_t>(k);
          const real_type   t           = h_G_[column_offset + row];
          h_G_[column_offset + row]     = h_c_[k] * t + h_s_[k] * h_G_[column_offset + row + 1];
          h_G_[column_offset + row + 1] = -h_s_[k] * t + h_c_[k] * h_G_[column_offset + row + 1];
        }

        const std::size_t diagonal_row = static_cast<std::size_t>(i);
        const real_type   diagonal     = h_G_[column_offset + diagonal_row];
        const real_type   next         = h_G_[column_offset + diagonal_row + 1];
        const real_type   gamma        = std::hypot(diagonal, next);
        if (gamma == ZERO || !std::isfinite(gamma))
        {
          out::error() << "GCRO-DR encountered a singular recycled least-squares problem.\n";
          return 1;
        }

        h_c_[i]      = diagonal / gamma;
        h_s_[i]      = next / gamma;
        h_rs_[i + 1] = -h_s_[i] * h_rs_[i];
        h_rs_[i] *= h_c_[i];
        h_G_[column_offset + diagonal_row]     = gamma;
        h_G_[column_offset + diagonal_row + 1] = ZERO;

        ++total_iters_;
        last_cycle_dim_ = i + 1;
        residual_norm   = std::abs(h_rs_[i + 1]);

        io::Logger::misc() << "GCRO-DR recycled it " << total_iters_
                           << ": estimated norm of residual "
                           << std::scientific << std::setprecision(16)
                           << residual_norm << "\n";

        if (hasConverged(residual_norm, rhs_norm) || happy_breakdown)
        {
          break;
        }
      }

      if (last_cycle_dim_ == 0)
      {
        out::error() << "GCRO-DR recycled cycle performed no Arnoldi iteration.\n";
        return 1;
      }

      // The augmented harmonic Ritz update uses B on the host. Transfer the
      // complete small block once per cycle instead of one scalar per basis
      // vector and Arnoldi iteration.
      {
        RESOLVE_PROFILE_SCOPE("GCRODR.cycle.B_to_host_sync");
        if (memspace_ == memory::DEVICE
            && vec_B_->copyToExternal(h_B_, memspace_, memory::HOST) != 0)
        {
          return 1;
        }
      }

      if (SmallDenseLinearAlgebra::triangularSolve('U',
                                                   'N',
                                                   'N',
                                                   last_cycle_dim_,
                                                   1,
                                                   h_G_,
                                                   restart_ + 1,
                                                   h_rs_,
                                                   restart_ + 1)
          != 0)
      {
        out::error() << "GCRO-DR failed to solve a recycled least-squares problem.\n";
        return 1;
      }

      // Apply the fixed right-preconditioned augmented Krylov correction
      //
      //   x <- x + (M^{-1} V_m - U B_m) y.
      //
      // The -U B_m y term cancels the component of A M^{-1} V_m y in
      // range(C), so the true residual agrees with the projected Arnoldi
      // least-squares residual.
      vec_work_->setToZero(memspace_);
      for (index_type j = 0; j < last_cycle_dim_; ++j)
      {
        vec_v.setData(vec_V_->getData(j, memspace_), memspace_);
        vector_handler_->axpy(h_rs_[j], &vec_v, vec_work_, memspace_);
      }
      if (preconditioner_->apply(vec_work_, vec_Z_) != 0)
      {
        out::error() << "GCRO-DR recycled correction preconditioner application failed.\n";
        return 1;
      }
      vector_handler_->axpy(ONE, vec_Z_, x, memspace_);
      if (memspace_ == memory::DEVICE)
      {
        std::fill(correction_coefficients.begin(),
                  correction_coefficients.end(),
                  ZERO);
        for (index_type j = 0; j < active_recycle_dim_; ++j)
        {
          for (index_type i = 0; i < last_cycle_dim_; ++i)
          {
            correction_coefficients[static_cast<std::size_t>(j)] +=
                h_B_[static_cast<std::size_t>(i)
                         * static_cast<std::size_t>(recycle_capacity)
                     + static_cast<std::size_t>(j)]
                * h_rs_[i];
          }
        }
        if (vec_projection_->copyFromExternal(correction_coefficients.data(),
                                              memory::HOST,
                                              memspace_)
            != 0)
        {
          return 1;
        }
        vector_handler_->gemv('N',
                              active_recycle_dim_,
                              MINUS_ONE,
                              ONE,
                              vec_U_,
                              vec_projection_,
                              x,
                              memspace_);
      }
      else
      {
        for (index_type j = 0; j < active_recycle_dim_; ++j)
        {
          real_type coefficient = ZERO;
          for (index_type i = 0; i < last_cycle_dim_; ++i)
          {
            coefficient += h_B_[static_cast<std::size_t>(i)
                                    * static_cast<std::size_t>(recycle_capacity)
                                + static_cast<std::size_t>(j)]
                           * h_rs_[i];
          }
          vec_u.setData(vec_U_->getData(j, memspace_), memspace_);
          vector_handler_->axpy(-coefficient, &vec_u, x, memspace_);
        }
      }

      {
        RESOLVE_PROFILE_SCOPE("GCRODR.cycle.true_residual_sync");
        rhs->copyToExternal(vec_work_->getData(memspace_), 0, memspace_, memspace_);
        if (matrix_handler_->matvec(A_, x, vec_work_, &MINUS_ONE, &ONE, memspace_) != 0)
        {
          out::error() << "GCRO-DR failed to compute a recycled-cycle true residual.\n";
          return 1;
        }
        final_norm = vector_handler_->dot(vec_work_, vec_work_, memspace_);
        final_norm = std::sqrt(std::max(ZERO, final_norm));
      }
      final_residual_norm_ = relativeResidual(final_norm, rhs_norm);

      io::Logger::misc() << "GCRO-DR recycled cycle: true norm of residual "
                         << std::scientific << std::setprecision(16)
                         << final_norm << "\n";

      if (updateRecycleSpace() != 0)
      {
        out::error() << "GCRO-DR failed to update the recycle space.\n";
        clearRecycleSpace();
        return 1;
      }
    }

    return 0;
  }

  bool LinSolverIterativeGCRODR::parametersValid() const
  {
    return !isNegative(restart_) && fitsLapackInt(restart_) && restart_ > 0
           && !isNegative(recycle_dim_) && fitsLapackInt(recycle_dim_)
           && recycle_dim_ < restart_
           && !isNegative(conv_cond_) && conv_cond_ <= 2
           && !isNegative(maxit_) && maxit_ > 0 && tol_ >= 0.0;
  }

  index_type LinSolverIterativeGCRODR::recycleCapacity() const
  {
    if (recycle_dim_ == 0 || restart_ <= 1)
    {
      return 0;
    }
    return std::min(recycle_dim_ + 1, restart_ - 1);
  }

  bool LinSolverIterativeGCRODR::hasConverged(real_type residual_norm,
                                              real_type rhs_norm) const
  {
    using namespace constants;

    switch (conv_cond_)
    {
    case 0:
      return residual_norm <= MACHINE_EPSILON;
    case 1:
      return residual_norm <= std::max(tol_, MACHINE_EPSILON);
    case 2:
      if (rhs_norm <= MACHINE_EPSILON)
      {
        return residual_norm <= std::max(tol_, MACHINE_EPSILON);
      }
      return residual_norm <= std::max(tol_ * rhs_norm, MACHINE_EPSILON);
    default:
      return false;
    }
  }

  real_type LinSolverIterativeGCRODR::relativeResidual(real_type residual_norm,
                                                       real_type rhs_norm) const
  {
    using namespace constants;
    return rhs_norm > MACHINE_EPSILON ? residual_norm / rhs_norm : residual_norm;
  }

  void LinSolverIterativeGCRODR::setMemorySpace()
  {
    handlers_compatible_ = false;
    if (matrix_handler_ == nullptr || vector_handler_ == nullptr)
    {
      out::error() << "GCRO-DR requires valid matrix and vector handlers.\n";
      return;
    }

    const bool matrix_cuda = matrix_handler_->getIsCudaEnabled();
    const bool matrix_hip  = matrix_handler_->getIsHipEnabled();
    const bool vector_cuda = vector_handler_->getIsCudaEnabled();
    const bool vector_hip  = vector_handler_->getIsHipEnabled();
    if (matrix_cuda != vector_cuda || matrix_hip != vector_hip)
    {
      out::error() << "Matrix and vector handler backends are incompatible.\n";
      return;
    }
    memspace_            = (matrix_cuda || matrix_hip) ? memory::DEVICE : memory::HOST;
    handlers_compatible_ = true;
  }

  void LinSolverIterativeGCRODR::initParamList()
  {
    params_list_["tol"]         = TOL;
    params_list_["maxit"]       = MAXIT;
    params_list_["restart"]     = RESTART;
    params_list_["conv_cond"]   = CONV_COND;
    params_list_["recycle_dim"] = RECYCLE_DIM;
  }
} // namespace ReSolve
