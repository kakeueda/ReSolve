/**
 * @file LinSolverIterativeGCRODR.hpp
 * @brief Declaration of the GCRO-DR iterative solver.
 */
#pragma once

#include <resolve/LinSolverIterative.hpp>
#include <resolve/MemoryUtils.hpp>

namespace ReSolve
{
  class GramSchmidt;

  /**
   * @brief GCRO-DR solver for sequences of slowly changing linear systems.
   *
   * The solver builds a recycle space from harmonic Ritz vectors after its
   * initial GMRES cycle, then updates that space after every projected Arnoldi
   * cycle. Fixed right preconditioning is supported. A valid recycle space can
   * be reused for later right-hand sides, and its transformed basis is
   * refreshed when a same-size matrix or preconditioner changes.
   */
  class LinSolverIterativeGCRODR : public LinSolverIterative
  {
    using vector_type = vector::Vector;

  public:
    LinSolverIterativeGCRODR(MatrixHandler* matrix_handler,
                             VectorHandler* vector_handler,
                             GramSchmidt*   gs);
    LinSolverIterativeGCRODR(index_type     restart,
                             index_type     recycle_dim,
                             real_type      tol,
                             index_type     maxit,
                             index_type     conv_cond,
                             MatrixHandler* matrix_handler,
                             VectorHandler* vector_handler,
                             GramSchmidt*   gs);
    ~LinSolverIterativeGCRODR();

    int solve(vector_type* rhs, vector_type* x) override;
    int setup(matrix::Sparse* A) override;
    int resetMatrix(matrix::Sparse* new_A) override;
    int setPreconditioner(Preconditioner* preconditioner) override;
    int setOrthogonalization(GramSchmidt* gs) override;

    int setRestart(index_type restart);
    int setRecycleDimension(index_type recycle_dim);
    int setConvergenceCondition(index_type conv_cond);
    int clearRecycleSpace();

    index_type         getRestart() const;
    index_type         getRecycleDimension() const;
    index_type         getActiveRecycleDimension() const;
    index_type         getConvCond() const;
    const vector_type* getRecycleBasis() const;
    const vector_type* getRecycleImage() const;

    int         setCliParam(const std::string id, const std::string value) override;
    std::string getCliParamString(const std::string id) const override;
    index_type  getCliParamInt(const std::string id) const override;
    real_type   getCliParamReal(const std::string id) const override;
    bool        getCliParamBool(const std::string id) const override;
    int         printCliParam(const std::string id) const override;

  private:
    enum ParameterIDs
    {
      TOL = 0,
      MAXIT,
      RESTART,
      CONV_COND,
      RECYCLE_DIM
    };

    int        allocateSolverData();
    int        freeSolverData();
    int        buildInitialRecycleSpace();
    int        updateRecycleSpace();
    int        refreshRecycleSpace();
    int        runRecycledCycles(vector_type* rhs,
                                 vector_type* x,
                                 real_type    rhs_norm,
                                 real_type&   final_norm);
    index_type recycleCapacity() const;
    bool       parametersValid() const;
    bool       hasConverged(real_type residual_norm, real_type rhs_norm) const;
    real_type  relativeResidual(real_type residual_norm, real_type rhs_norm) const;
    void       setMemorySpace();
    void       initParamList();

    index_type restart_{30};
    index_type recycle_dim_{10};
    index_type active_recycle_dim_{0};
    index_type conv_cond_{2};

    memory::MemorySpace memspace_{memory::HOST};

    vector_type* vec_U_{nullptr};
    vector_type* vec_C_{nullptr};
    vector_type* vec_V_{nullptr};
    vector_type* vec_Z_{nullptr};
    vector_type* vec_work_{nullptr};
    vector_type* vec_Y_{nullptr};
    vector_type* vec_B_{nullptr};
    vector_type* vec_projection_{nullptr};
    vector_type* vec_augmented_image_{nullptr};
    vector_type* vec_augmented_domain_{nullptr};
    vector_type* vec_augmented_product_{nullptr};
    vector_type* vec_image_coefficients_{nullptr};

    real_type* h_H_{nullptr};
    real_type* h_B_{nullptr};
    real_type* h_G_{nullptr};
    real_type* h_c_{nullptr};
    real_type* h_s_{nullptr};
    real_type* h_rs_{nullptr};

    GramSchmidt* GS_{nullptr};
    index_type   n_{0};
    index_type   last_cycle_dim_{0};
    bool         is_solver_set_{false};
    bool         recycle_image_valid_{false};
    bool         handlers_compatible_{false};
  };
} // namespace ReSolve
