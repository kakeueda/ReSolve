/**
 * @file GCRODRTests.hpp
 * @brief Unit tests for GCRO-DR setup, recycle construction, and cycles.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeGCRODR.hpp>
#include <resolve/Preconditioner.hpp>
#include <resolve/PreconditionerUserMatrix.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    class IdentityPreconditioner : public Preconditioner
    {
    public:
      explicit IdentityPreconditioner(memory::MemorySpace memspace)
        : memspace_(memspace)
      {
      }

      int apply(vector_type* rhs, vector_type* x) override
      {
        return x->copyFromExternal(rhs, memspace_, memspace_);
      }

      int setup(matrix_type* /* A */) override
      {
        return 0;
      }

    private:
      memory::MemorySpace memspace_;
    };

    class GCRODRTests : public TestBase
    {
    public:
      TestOutcome initialCycleConverges()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        matrix::Csr      A(4, 4, 10);
        const index_type rows[]   = {0, 2, 5, 8, 10};
        const index_type cols[]   = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
        const real_type  values[] = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(4);
        vector::Vector x(4);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type rhs_values[] = {2.0, 8.0, -6.5, 2.5};
        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);

        const real_type          solver_tolerance = 100.0 * constants::MACHINE_EPSILON;
        LinSolverIterativeGCRODR solver(4,
                                        2,
                                        solver_tolerance,
                                        4,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);
        success *= preconditioner.setup(&A) == 0;
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;

        const real_type expected[]  = {1.0, 2.0, -1.0, 0.5};
        const real_type error_bound = 5000.0 * constants::MACHINE_EPSILON;
        for (index_type i = 0; i < 4; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i] - expected[i]) <= error_bound;
        }
        success *= std::abs(solver.getInitResidualNorm() - 1.0) <= error_bound;
        success *= solver.getFinalResidualNorm() <= error_bound;
        success *= solver.getNumIter() > 0 && solver.getNumIter() <= 4;
        success *= solver.getActiveRecycleDimension() == 2;

        return success.report(__func__);
      }

      TestOutcome happyBreakdown()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::MGS);

        matrix::Csr      A(3, 3, 3);
        const index_type rows[]   = {0, 1, 2, 3};
        const index_type cols[]   = {0, 1, 2};
        const real_type  values[] = {1.0, 1.0, 1.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(3);
        vector::Vector x(3);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type rhs_values[] = {1.0, -2.0, 3.0};
        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(3,
                                        1,
                                        100.0 * constants::MACHINE_EPSILON,
                                        3,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getNumIter() == 1;
        success *= solver.getActiveRecycleDimension() == 1;

        const real_type error_bound = 100.0 * constants::MACHINE_EPSILON;
        for (index_type i = 0; i < 3; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i] - rhs_values[i]) <= error_bound;
        }
        success *= solver.getFinalResidualNorm() <= error_bound;

        return success.report(__func__);
      }

      TestOutcome recycleSpaceProperties()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        matrix::Csr      A(4, 4, 4);
        const index_type rows[]   = {0, 1, 2, 3, 4};
        const index_type cols[]   = {0, 1, 2, 3};
        const real_type  values[] = {1.0, 2.0, 4.0, 8.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        matrix::Csr     inverse_preconditioner(4, 4, 4);
        const real_type inverse_values[] = {1.0,
                                            0.75,
                                            static_cast<real_type>(0.6),
                                            static_cast<real_type>(0.4)};
        inverse_preconditioner.allocateAll(memory::HOST);
        inverse_preconditioner.copyFromExternal(rows,
                                                cols,
                                                inverse_values,
                                                memory::HOST,
                                                memory::HOST);

        vector::Vector rhs(4);
        vector::Vector x(4);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        rhs.setToConst(1.0, memory::HOST);
        x.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(4,
                                        2,
                                        100.0 * constants::MACHINE_EPSILON,
                                        4,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        PreconditionerUserMatrix preconditioner(&matrix_handler);
        success *= preconditioner.setPrecMatrix(&inverse_preconditioner) == 0;
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getActiveRecycleDimension() == 2;

        const vector::Vector* U = solver.getRecycleBasis();
        const vector::Vector* C = solver.getRecycleImage();
        success *= U != nullptr && C != nullptr;

        const real_type error_bound = 10000.0 * constants::MACHINE_EPSILON;
        if (U != nullptr && C != nullptr)
        {
          // C must be orthonormal.
          for (index_type column = 0; column < 2; ++column)
          {
            for (index_type row = 0; row < 2; ++row)
            {
              real_type dot = 0.0;
              for (index_type i = 0; i < 4; ++i)
              {
                dot += C->getData(column, memory::HOST)[i]
                       * C->getData(row, memory::HOST)[i];
              }
              const real_type expected = row == column ? 1.0 : 0.0;
              success *= std::abs(dot - expected) <= error_bound;
            }
          }

          // The two smallest harmonic Ritz values of the right-preconditioned
          // operator are 1 and 1.5, so range(C) should be span(e_1,e_2).
          real_type unwanted_component = 0.0;
          for (index_type column = 0; column < 2; ++column)
          {
            unwanted_component += C->getData(column, memory::HOST)[2]
                                  * C->getData(column, memory::HOST)[2];
            unwanted_component += C->getData(column, memory::HOST)[3]
                                  * C->getData(column, memory::HOST)[3];
          }
          success *= std::sqrt(unwanted_component) <= error_bound;

          // The recycle matrices must satisfy A U = C.
          vector::Vector u(4);
          vector::Vector Au(4);
          u.allocate(memory::HOST);
          Au.allocate(memory::HOST);
          for (index_type column = 0; column < 2; ++column)
          {
            u.copyFromExternal(U->getData(column, memory::HOST),
                               memory::HOST,
                               memory::HOST);
            Au.setToZero(memory::HOST);
            success *= matrix_handler.matvec(&A,
                                             &u,
                                             &Au,
                                             &constants::ONE,
                                             &constants::ZERO,
                                             memory::HOST)
                       == 0;
            real_type error = 0.0;
            for (index_type i = 0; i < 4; ++i)
            {
              const real_type difference = Au.getData(memory::HOST)[i]
                                           - C->getData(column, memory::HOST)[i];
              error += difference * difference;
            }
            success *= std::sqrt(error) <= error_bound;
          }
        }

        return success.report(__func__);
      }

      TestOutcome recycledCyclesConverge()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        constexpr index_type n = 6;
        matrix::Csr          A(n, n, n);
        const index_type     rows[]   = {0, 1, 2, 3, 4, 5, 6};
        const index_type     cols[]   = {0, 1, 2, 3, 4, 5};
        const real_type      values[] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(n);
        vector::Vector x(n);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        rhs.setToConst(1.0, memory::HOST);
        x.setToZero(memory::HOST);

        constexpr index_type     restart   = 4;
        constexpr index_type     recycle   = 2;
        constexpr index_type     maxit     = 100;
        const real_type          tolerance = std::max(static_cast<real_type>(1.0e-7),
                                             static_cast<real_type>(1000.0)
                                                 * constants::MACHINE_EPSILON);
        LinSolverIterativeGCRODR solver(restart,
                                        recycle,
                                        tolerance,
                                        maxit,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;

        // The first cycle cannot solve a six-eigenvalue problem with four
        // Arnoldi vectors. Convergence therefore exercises at least one
        // projected cycle with the active recycle space.
        success *= solver.getNumIter() > restart;
        success *= solver.getNumIter() < maxit;
        success *= solver.getActiveRecycleDimension() == recycle;
        success *= solver.getFinalResidualNorm()
                   <= static_cast<real_type>(2.0) * tolerance;

        const real_type error_bound = static_cast<real_type>(100.0) * tolerance;
        for (index_type i = 0; i < n; ++i)
        {
          const real_type expected = constants::ONE / values[i];
          success *= std::abs(x.getData(memory::HOST)[i] - expected) <= error_bound;
        }

        // A second right-hand side starts directly from the existing U,C
        // pair instead of rebuilding it with an unrecycled initial cycle.
        const real_type rhs_values[] = {2.0, -2.0, 1.0, 4.0, -1.0, 3.0};
        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getNumIter() > 0;
        success *= solver.getNumIter() < maxit;
        success *= solver.getActiveRecycleDimension() == recycle;
        success *= solver.getFinalResidualNorm()
                   <= static_cast<real_type>(2.0) * tolerance;
        for (index_type i = 0; i < n; ++i)
        {
          const real_type expected = rhs_values[i] / values[i];
          success *= std::abs(x.getData(memory::HOST)[i] - expected) <= error_bound;
        }

        return success.report(__func__);
      }

      TestOutcome recycleSpaceIsUpdated()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        constexpr index_type n       = 6;
        constexpr index_type restart = 4;
        constexpr index_type recycle = 2;
        matrix::Csr          A(n, n, n);
        matrix::Csr          inverse_preconditioner(n, n, n);
        const index_type     rows[]           = {0, 1, 2, 3, 4, 5, 6};
        const index_type     cols[]           = {0, 1, 2, 3, 4, 5};
        const real_type      values[]         = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
        const real_type      inverse_values[] = {1.0,
                                                 0.75,
                                                 static_cast<real_type>(0.6),
                                                 static_cast<real_type>(0.4),
                                                 static_cast<real_type>(0.3),
                                                 static_cast<real_type>(0.2)};
        A.allocateAll(memory::HOST);
        inverse_preconditioner.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        inverse_preconditioner.copyFromExternal(rows,
                                                cols,
                                                inverse_values,
                                                memory::HOST,
                                                memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(n);
        vector::Vector x(n);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        rhs.setToConst(1.0, memory::HOST);
        x.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(restart,
                                        recycle,
                                        100.0 * constants::MACHINE_EPSILON,
                                        restart,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        PreconditionerUserMatrix preconditioner(&matrix_handler);
        success *= preconditioner.setPrecMatrix(&inverse_preconditioner) == 0;
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getActiveRecycleDimension() == recycle;

        std::vector<real_type> initial_C(static_cast<std::size_t>(n * recycle));
        const vector::Vector*  C = solver.getRecycleImage();
        success *= C != nullptr;
        if (C != nullptr)
        {
          for (index_type column = 0; column < recycle; ++column)
          {
            std::copy_n(C->getData(column, memory::HOST),
                        static_cast<std::size_t>(n),
                        initial_C.data() + static_cast<std::size_t>(column * n));
          }
        }

        // Exactly one projected cycle updates the harmonic Ritz space.
        const real_type rhs_values[] = {2.0, -1.0, 3.0, 0.5, -2.0, 4.0};
        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        solver.setMaxit(restart - recycle);
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getNumIter() == restart - recycle;
        success *= solver.getActiveRecycleDimension() == recycle;

        const vector::Vector* U = solver.getRecycleBasis();
        C                       = solver.getRecycleImage();
        success *= U != nullptr && C != nullptr;
        const real_type error_bound = static_cast<real_type>(20000.0)
                                      * constants::MACHINE_EPSILON;
        if (U != nullptr && C != nullptr)
        {
          // Compare the old and new orthogonal projectors. A nonzero distance
          // proves that the cycle did not leave the initial space fixed.
          real_type overlap_squared = 0.0;
          for (index_type old_column = 0; old_column < recycle; ++old_column)
          {
            for (index_type new_column = 0; new_column < recycle; ++new_column)
            {
              real_type dot = 0.0;
              for (index_type i = 0; i < n; ++i)
              {
                dot += initial_C[static_cast<std::size_t>(old_column * n + i)]
                       * C->getData(new_column, memory::HOST)[i];
              }
              overlap_squared += dot * dot;
            }
          }
          const real_type projector_distance =
              std::sqrt(std::max(static_cast<real_type>(0.0),
                                 static_cast<real_type>(2 * recycle)
                                     - static_cast<real_type>(2.0) * overlap_squared));
          success *= projector_distance
                     > static_cast<real_type>(100.0) * constants::MACHINE_EPSILON;

          vector::Vector u(n);
          vector::Vector Au(n);
          u.allocate(memory::HOST);
          Au.allocate(memory::HOST);
          for (index_type column = 0; column < recycle; ++column)
          {
            for (index_type row = 0; row < recycle; ++row)
            {
              real_type dot = 0.0;
              for (index_type i = 0; i < n; ++i)
              {
                dot += C->getData(column, memory::HOST)[i]
                       * C->getData(row, memory::HOST)[i];
              }
              const real_type expected = row == column ? 1.0 : 0.0;
              success *= std::abs(dot - expected) <= error_bound;
            }

            u.copyFromExternal(U->getData(column, memory::HOST),
                               memory::HOST,
                               memory::HOST);
            Au.setToZero(memory::HOST);
            success *= matrix_handler.matvec(&A,
                                             &u,
                                             &Au,
                                             &constants::ONE,
                                             &constants::ZERO,
                                             memory::HOST)
                       == 0;
            real_type relation_error = 0.0;
            for (index_type i = 0; i < n; ++i)
            {
              const real_type difference = Au.getData(memory::HOST)[i]
                                           - C->getData(column, memory::HOST)[i];
              relation_error += difference * difference;
            }
            success *= std::sqrt(relation_error) <= error_bound;
          }
        }

        // A matrix change refreshes the retained transformed basis instead of
        // discarding it. Verify the refreshed relation before and after solve.
        matrix::Csr     updated_A(n, n, n);
        const real_type updated_values[] = {static_cast<real_type>(1.2),
                                            static_cast<real_type>(1.8),
                                            4.5,
                                            7.5,
                                            17.0,
                                            30.0};
        updated_A.allocateAll(memory::HOST);
        updated_A.copyFromExternal(rows,
                                   cols,
                                   updated_values,
                                   memory::HOST,
                                   memory::HOST);
        success *= solver.resetMatrix(&updated_A) == 0;
        success *= solver.getActiveRecycleDimension() == recycle;
        U = solver.getRecycleBasis();
        C = solver.getRecycleImage();
        success *= U != nullptr && C != nullptr;
        if (U != nullptr && C != nullptr)
        {
          vector::Vector u(n);
          vector::Vector Au(n);
          u.allocate(memory::HOST);
          Au.allocate(memory::HOST);
          for (index_type column = 0; column < recycle; ++column)
          {
            u.copyFromExternal(U->getData(column, memory::HOST),
                               memory::HOST,
                               memory::HOST);
            Au.setToZero(memory::HOST);
            success *= matrix_handler.matvec(&updated_A,
                                             &u,
                                             &Au,
                                             &constants::ONE,
                                             &constants::ZERO,
                                             memory::HOST)
                       == 0;
            real_type relation_error = 0.0;
            for (index_type i = 0; i < n; ++i)
            {
              const real_type difference = Au.getData(memory::HOST)[i]
                                           - C->getData(column, memory::HOST)[i];
              relation_error += difference * difference;
            }
            success *= std::sqrt(relation_error) <= error_bound;
          }
        }

        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        const real_type solve_tolerance =
            std::max(static_cast<real_type>(1.0e-7),
                     static_cast<real_type>(1000.0) * constants::MACHINE_EPSILON);
        solver.setTol(solve_tolerance);
        solver.setMaxit(100);
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getFinalResidualNorm()
                   <= static_cast<real_type>(2.0) * solve_tolerance;
        for (index_type i = 0; i < n; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i]
                              - rhs_values[i] / updated_values[i])
                     <= static_cast<real_type>(100.0) * solve_tolerance;
        }

        return success.report(__func__);
      }

      TestOutcome complexPairSelection()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        // Eigenvalues are +i, -i, and 5. With recycle_dim=1 the conjugate
        // pair must not be split, so GCRO-DR retains both vectors (k+1).
        matrix::Csr      A(3, 3, 3);
        const index_type rows[]   = {0, 1, 2, 3};
        const index_type cols[]   = {1, 0, 2};
        const real_type  values[] = {-1.0, 1.0, 5.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(3);
        vector::Vector x(3);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type rhs_values[] = {1.0, 2.0, 3.0};
        rhs.copyFromExternal(rhs_values, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(3,
                                        1,
                                        100.0 * constants::MACHINE_EPSILON,
                                        3,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getActiveRecycleDimension() == 2;

        const vector::Vector* C = solver.getRecycleImage();
        success *= C != nullptr;
        if (C != nullptr)
        {
          const real_type error_bound                = 10000.0 * constants::MACHINE_EPSILON;
          real_type       real_eigenvector_component = 0.0;
          for (index_type column = 0; column < 2; ++column)
          {
            real_eigenvector_component += C->getData(column, memory::HOST)[2]
                                          * C->getData(column, memory::HOST)[2];
          }
          success *= std::sqrt(real_eigenvector_component) <= error_bound;
        }

        // Update the augmented harmonic Ritz space after one projected
        // cycle. The selected +i/-i pair must remain intact.
        const real_type second_rhs[] = {-2.0, 1.0, 4.0};
        rhs.copyFromExternal(second_rhs, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        solver.setMaxit(1);
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getNumIter() == 1;
        success *= solver.getActiveRecycleDimension() == 2;
        C = solver.getRecycleImage();
        success *= C != nullptr;
        if (C != nullptr)
        {
          const real_type error_bound                = 20000.0 * constants::MACHINE_EPSILON;
          real_type       real_eigenvector_component = 0.0;
          for (index_type column = 0; column < 2; ++column)
          {
            real_eigenvector_component += C->getData(column, memory::HOST)[2]
                                          * C->getData(column, memory::HOST)[2];
          }
          success *= std::sqrt(real_eigenvector_component) <= error_bound;
        }

        const real_type expected[] = {1.0, 2.0, static_cast<real_type>(0.8)};
        for (index_type i = 0; i < 3; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i] - expected[i])
                     <= 20000.0 * constants::MACHINE_EPSILON;
        }

        return success.report(__func__);
      }

      TestOutcome rightPreconditionedUpdateUsesTransformedBasis()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        matrix::Csr      A(3, 3, 3);
        const index_type rows_A[]   = {0, 1, 2, 3};
        const index_type cols_A[]   = {0, 1, 2};
        const real_type  values_A[] = {1.0, 4.0, 10.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows_A, cols_A, values_A, memory::HOST, memory::HOST);

        // M^{-1} is lower triangular and does not commute with A. Thus
        // T=A M^{-1} has a smallest-eigenvalue eigenvector proportional to
        // (3,-4,0), whereas A's corresponding eigenvector is (1,0,0).
        matrix::Csr      inverse_preconditioner(3, 3, 4);
        const index_type rows_M[]   = {0, 1, 3, 4};
        const index_type cols_M[]   = {0, 0, 1, 2};
        const real_type  values_M[] = {1.0, 1.0, 1.0, 1.0};
        inverse_preconditioner.allocateAll(memory::HOST);
        inverse_preconditioner.copyFromExternal(rows_M,
                                                cols_M,
                                                values_M,
                                                memory::HOST,
                                                memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(3);
        vector::Vector x(3);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type first_rhs[] = {1.0, 2.0, 3.0};
        rhs.copyFromExternal(first_rhs, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(3,
                                        1,
                                        100.0 * constants::MACHINE_EPSILON,
                                        3,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        PreconditionerUserMatrix preconditioner(&matrix_handler);
        success *= preconditioner.setPrecMatrix(&inverse_preconditioner) == 0;
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getActiveRecycleDimension() == 1;

        const real_type second_rhs[] = {-2.0, 1.0, 4.0};
        rhs.copyFromExternal(second_rhs, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        solver.setMaxit(2);
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getActiveRecycleDimension() == 1;

        const vector::Vector* C = solver.getRecycleImage();
        success *= C != nullptr;
        if (C != nullptr)
        {
          const real_type alignment =
              (static_cast<real_type>(3.0) * C->getData(0, memory::HOST)[0]
               - static_cast<real_type>(4.0) * C->getData(0, memory::HOST)[1])
              / static_cast<real_type>(5.0);
          success *= std::abs(std::abs(alignment) - constants::ONE)
                     <= static_cast<real_type>(20000.0)
                            * constants::MACHINE_EPSILON;
          success *= std::abs(C->getData(0, memory::HOST)[2])
                     <= static_cast<real_type>(20000.0)
                            * constants::MACHINE_EPSILON;
        }

        return success.report(__func__);
      }

      TestOutcome systemSolverSequence()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();

        matrix::Csr      A1(4, 4, 4);
        matrix::Csr      A2(4, 4, 4);
        const index_type rows[]    = {0, 1, 2, 3, 4};
        const index_type cols[]    = {0, 1, 2, 3};
        const real_type  values1[] = {1.0, 2.0, 4.0, 8.0};
        const real_type  values2[] = {1.5, 2.5, 3.5, 7.0};
        A1.allocateAll(memory::HOST);
        A2.allocateAll(memory::HOST);
        A1.copyFromExternal(rows, cols, values1, memory::HOST, memory::HOST);
        A2.copyFromExternal(rows, cols, values2, memory::HOST, memory::HOST);

        vector::Vector rhs(4);
        vector::Vector x(4);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type first_rhs[] = {1.0, -2.0, 3.0, 4.0};
        rhs.copyFromExternal(first_rhs, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);

        SystemSolver system_solver(&workspace,
                                   "none",
                                   "none",
                                   "gcrodr",
                                   "ilu0",
                                   "none");
        success *= system_solver.getSolveMethod() == "gcrodr";
        auto* gcrodr = dynamic_cast<LinSolverIterativeGCRODR*>(
            &system_solver.getIterativeSolver());
        success *= gcrodr != nullptr;
        if (gcrodr == nullptr)
        {
          return success.report(__func__);
        }

        success *= gcrodr->setCliParam("recycle_dim", "1") == 0;
        success *= gcrodr->setCliParam("restart", "3") == 0;
        success *= gcrodr->setCliParam("maxit", "20") == 0;
        success *= gcrodr->setCliParam("tol", "1e-7") == 0;
        success *= system_solver.setMatrix(&A1) == 0;
        success *= system_solver.preconditionerSetup("left") != 0;
        success *= system_solver.preconditionerSetup("right") == 0;
        success *= system_solver.solve(&rhs, &x) == 0;
        success *= gcrodr->getActiveRecycleDimension() == 1;

        const real_type error_bound = static_cast<real_type>(1.0e-5);
        for (index_type i = 0; i < 4; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i]
                              - first_rhs[i] / values1[i])
                     <= error_bound;
        }

        // Change both the matrix and its ILU0 preconditioner. The explicit
        // reset path must refresh and reuse the GCRO-DR space.
        const real_type second_rhs[] = {-1.0, 3.0, 2.0, -4.0};
        rhs.copyFromExternal(second_rhs, memory::HOST, memory::HOST);
        x.setToZero(memory::HOST);
        success *= system_solver.setMatrix(&A2) == 0;
        success *= system_solver.resetPreconditioner(&A2) == 0;
        success *= system_solver.solve(&rhs, &x) == 0;
        success *= gcrodr->getActiveRecycleDimension() == 1;
        success *= gcrodr->getFinalResidualNorm() <= error_bound;
        for (index_type i = 0; i < 4; ++i)
        {
          success *= std::abs(x.getData(memory::HOST)[i]
                              - second_rhs[i] / values2[i])
                     <= error_bound;
        }

        SystemSolver switched_solver(&workspace,
                                     "none",
                                     "none",
                                     "fgmres",
                                     "none",
                                     "none");
        success *= switched_solver.setSolveMethod("gcrodr") == 0;
        success *= dynamic_cast<LinSolverIterativeGCRODR*>(
                       &switched_solver.getIterativeSolver())
                   != nullptr;

        return success.report(__func__);
      }

      TestOutcome zeroInitialResidual()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        matrix::Csr      A(2, 2, 2);
        const index_type rows[]   = {0, 1, 2};
        const index_type cols[]   = {0, 1};
        const real_type  values[] = {1.0, 1.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);
        matrix_handler.setValuesChanged(true, memory::HOST);

        vector::Vector rhs(2);
        vector::Vector x(2);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        const real_type values_x[] = {2.0, -1.0};
        rhs.copyFromExternal(values_x, memory::HOST, memory::HOST);
        x.copyFromExternal(values_x, memory::HOST, memory::HOST);

        LinSolverIterativeGCRODR solver(2,
                                        1,
                                        100.0 * constants::MACHINE_EPSILON,
                                        2,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);
        success *= solver.setup(&A) == 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&rhs, &x) == 0;
        success *= solver.getNumIter() == 0;
        success *= solver.getInitResidualNorm() == 0.0;
        success *= solver.getFinalResidualNorm() == 0.0;

        return success.report(__func__);
      }

      TestOutcome invalidSolveData()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();
        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::MGS);

        matrix::Csr      A(2, 2, 2);
        const index_type rows[]   = {0, 1, 2};
        const index_type cols[]   = {0, 1};
        const real_type  values[] = {1.0, 1.0};
        A.allocateAll(memory::HOST);
        A.copyFromExternal(rows, cols, values, memory::HOST, memory::HOST);

        vector::Vector rhs(2);
        vector::Vector x(2);
        vector::Vector wrong_size(3);
        rhs.allocate(memory::HOST);
        x.allocate(memory::HOST);
        wrong_size.allocate(memory::HOST);
        rhs.setToConst(1.0, memory::HOST);
        x.setToZero(memory::HOST);
        wrong_size.setToZero(memory::HOST);

        LinSolverIterativeGCRODR solver(2,
                                        1,
                                        100.0 * constants::MACHINE_EPSILON,
                                        2,
                                        2,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);
        IdentityPreconditioner   preconditioner(memory::HOST);

        success *= solver.solve(&rhs, &x) != 0;
        success *= solver.setup(&A) == 0;
        success *= solver.solve(&rhs, &x) != 0;
        success *= solver.setPreconditioner(&preconditioner) == 0;
        success *= solver.solve(&wrong_size, &x) != 0;
        success *= preconditioner.setSide(Preconditioner::Side::LEFT) == 0;
        success *= solver.solve(&rhs, &x) != 0;

        return success.report(__func__);
      }
    };
  } // namespace tests
} // namespace ReSolve
