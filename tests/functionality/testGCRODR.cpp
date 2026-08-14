/**
 * @file testGCRODR.cpp
 * @brief Functionality tests for GCRO-DR on Matrix Market systems.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <resolve/LinSolverIterativeGCRODR.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>

#ifdef RESOLVE_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace
{
  using ReSolve::index_type;
  using ReSolve::real_type;
  using MatrixType = ReSolve::matrix::Csr;
  using VectorType = ReSolve::vector::Vector;

  struct MatrixCase
  {
    const char* name;
    const char* matrix_1;
    const char* rhs_1;
    const char* matrix_2;
    const char* rhs_2;
  };

  const MatrixCase MATRIX_CASES[] = {
      {"spd", "A_spd01.mtx", "b_spd01.mtx", "A_spd02.mtx", "b_spd02.mtx"},
      {"symmetric",
       "A_symmetric01.mtx",
       "b_symmetric01.mtx",
       "A_symmetric02.mtx",
       "b_symmetric02.mtx"},
      {"asymmetric",
       "A_asymmetric01.mtx",
       "b_asymmetric01.mtx",
       "A_asymmetric02.mtx",
       "b_asymmetric02.mtx"}};

  const MatrixCase* findMatrixCase(const std::string& name)
  {
    for (const MatrixCase& matrix_case : MATRIX_CASES)
    {
      if (name == matrix_case.name)
      {
        return &matrix_case;
      }
    }
    return nullptr;
  }

  std::unique_ptr<MatrixType> loadMatrix(const std::string& path)
  {
    std::ifstream input(path);
    if (!input.is_open())
    {
      std::cout << "Failed to open matrix file " << path << "\n";
      return std::unique_ptr<MatrixType>();
    }
    return std::unique_ptr<MatrixType>(ReSolve::io::createCsrFromFile(input, true));
  }

  std::unique_ptr<VectorType> loadVector(const std::string& path)
  {
    std::ifstream input(path);
    if (!input.is_open())
    {
      std::cout << "Failed to open RHS file " << path << "\n";
      return std::unique_ptr<VectorType>();
    }
    return std::unique_ptr<VectorType>(ReSolve::io::createVectorFromFile(input));
  }

  int moveToMemorySpace(MatrixType*                  A,
                        VectorType*                  rhs_1,
                        VectorType*                  rhs_2,
                        ReSolve::memory::MemorySpace memspace)
  {
    if (memspace == ReSolve::memory::HOST)
    {
      return 0;
    }

    int status = A->allocateMatrixData(memspace);
    status += A->syncData(memspace);
    status += rhs_1->allocate(memspace);
    status += rhs_1->syncData(memspace);
    status += rhs_2->allocate(memspace);
    status += rhs_2->syncData(memspace);
    return status;
  }

  real_type computeRelativeResidual(ReSolve::MatrixHandler&      matrix_handler,
                                    ReSolve::VectorHandler&      vector_handler,
                                    MatrixType*                  A,
                                    VectorType*                  rhs,
                                    VectorType*                  x,
                                    ReSolve::memory::MemorySpace memspace)
  {
    using namespace ReSolve::constants;

    VectorType residual(rhs->getSize());
    residual.allocate(memspace);
    residual.copyFromExternal(rhs, memspace, memspace);
    matrix_handler.setValuesChanged(true, memspace);
    if (matrix_handler.matvec(A,
                              x,
                              &residual,
                              &MINUS_ONE,
                              &ONE,
                              memspace)
        != 0)
    {
      return std::numeric_limits<real_type>::infinity();
    }

    const real_type residual_norm =
        std::sqrt(std::max(ZERO,
                           vector_handler.dot(&residual, &residual, memspace)));
    const real_type rhs_norm =
        std::sqrt(std::max(ZERO, vector_handler.dot(rhs, rhs, memspace)));
    return rhs_norm == ZERO ? residual_norm : residual_norm / rhs_norm;
  }

  int solveAndCheck(const std::string&                 backend,
                    const std::string&                 phase,
                    ReSolve::SystemSolver&             solver,
                    ReSolve::MatrixHandler&            matrix_handler,
                    ReSolve::VectorHandler&            vector_handler,
                    ReSolve::LinSolverIterativeGCRODR& gcrodr,
                    MatrixType*                        A,
                    VectorType*                        rhs,
                    VectorType*                        x,
                    ReSolve::memory::MemorySpace       memspace,
                    real_type                          acceptance_tolerance)
  {
    int error_sum = 0;
    x->setToZero(memspace);
    const int status = solver.solve(rhs, x);
    if (status != 0)
    {
      std::cout << backend << " " << phase
                << ": solve failed with status " << status << "\n";
      return 1;
    }

    const real_type  measured_residual = computeRelativeResidual(matrix_handler,
                                                                vector_handler,
                                                                A,
                                                                rhs,
                                                                x,
                                                                memspace);
    const real_type  reported_residual = gcrodr.getFinalResidualNorm();
    const index_type active_recycle    = gcrodr.getActiveRecycleDimension();
    const index_type recycle_capacity =
        std::min(gcrodr.getRecycleDimension() + 1, gcrodr.getRestart() - 1);

    if (!std::isfinite(measured_residual)
        || measured_residual > acceptance_tolerance)
    {
      std::cout << backend << " " << phase
                << ": measured relative residual is too large: "
                << measured_residual << "\n";
      ++error_sum;
    }
    if (!std::isfinite(reported_residual)
        || reported_residual > acceptance_tolerance)
    {
      std::cout << backend << " " << phase
                << ": reported relative residual is too large: "
                << reported_residual << "\n";
      ++error_sum;
    }
    if (std::abs(measured_residual - reported_residual)
        > acceptance_tolerance)
    {
      std::cout << backend << " " << phase
                << ": reported and measured residuals disagree\n";
      ++error_sum;
    }
    if (gcrodr.getNumIter() == 0 || active_recycle == 0
        || active_recycle > recycle_capacity)
    {
      std::cout << backend << " " << phase
                << ": invalid iteration or recycle-space dimensions\n";
      ++error_sum;
    }

    std::cout << std::scientific << std::setprecision(6)
              << backend << " " << phase
              << ": residual=" << measured_residual
              << ", iterations=" << gcrodr.getNumIter()
              << ", active_recycle_dim=" << active_recycle << "\n";
    return error_sum;
  }

  template <class WorkspaceType>
  int runMatrixSequence(const std::string& data_path,
                        const MatrixCase&  matrix_case,
                        const std::string& backend)
  {
    using namespace ReSolve;

    WorkspaceType workspace;
    workspace.initializeHandles();

    MatrixHandler             matrix_handler(&workspace);
    VectorHandler             vector_handler(&workspace);
    const memory::MemorySpace memspace =
        matrix_handler.getIsCudaEnabled() ? memory::DEVICE : memory::HOST;

    std::unique_ptr<MatrixType> A =
        loadMatrix(data_path + "/" + matrix_case.matrix_1);
    std::unique_ptr<VectorType> rhs_1 =
        loadVector(data_path + "/" + matrix_case.rhs_1);
    std::unique_ptr<VectorType> rhs_2 =
        loadVector(data_path + "/" + matrix_case.rhs_2);
    if (!A || !rhs_1 || !rhs_2
        || A->getNumRows() != rhs_1->getSize()
        || A->getNumRows() != rhs_2->getSize())
    {
      std::cout << "Invalid input data for case " << matrix_case.name << "\n";
      return 1;
    }
    if (moveToMemorySpace(A.get(), rhs_1.get(), rhs_2.get(), memspace) != 0)
    {
      std::cout << "Failed to move " << matrix_case.name
                << " data to the " << backend << " memory space\n";
      return 1;
    }

    SystemSolver solver(&workspace, "none", "none", "gcrodr", "ilu0", "none");
    if (solver.setGramSchmidtMethod("cgs2") != 0)
    {
      return 1;
    }

    auto* gcrodr = dynamic_cast<LinSolverIterativeGCRODR*>(
        &solver.getIterativeSolver());
    if (gcrodr == nullptr)
    {
      std::cout << "SystemSolver did not create GCRO-DR\n";
      return 1;
    }

    const real_type solver_tolerance =
        std::max(static_cast<real_type>(1.0e-10),
                 static_cast<real_type>(100.0)
                     * std::numeric_limits<real_type>::epsilon());
    const real_type acceptance_tolerance =
        std::max(static_cast<real_type>(1.0e-8),
                 static_cast<real_type>(1000.0)
                     * std::numeric_limits<real_type>::epsilon());

    int error_sum = gcrodr->setRestart(30);
    error_sum += gcrodr->setRecycleDimension(4);
    gcrodr->setMaxit(400);
    gcrodr->setTol(solver_tolerance);

    error_sum += solver.setMatrix(A.get());
    error_sum += solver.preconditionerSetup("right");
    if (error_sum != 0)
    {
      std::cout << "Failed to set up GCRO-DR for " << matrix_case.name << "\n";
      return error_sum;
    }

    VectorType x(A->getNumRows());
    x.allocate(memspace);
    error_sum += solveAndCheck(backend,
                               matrix_case.name + std::string("/initial"),
                               solver,
                               matrix_handler,
                               vector_handler,
                               *gcrodr,
                               A.get(),
                               rhs_1.get(),
                               &x,
                               memspace,
                               acceptance_tolerance);

    const index_type first_recycle_dim = gcrodr->getActiveRecycleDimension();
    error_sum += solveAndCheck(backend,
                               matrix_case.name + std::string("/new-rhs"),
                               solver,
                               matrix_handler,
                               vector_handler,
                               *gcrodr,
                               A.get(),
                               rhs_2.get(),
                               &x,
                               memspace,
                               acceptance_tolerance);

    const index_type retained_recycle_dim = gcrodr->getActiveRecycleDimension();
    std::ifstream    matrix_update(data_path + "/" + matrix_case.matrix_2);
    if (!matrix_update.is_open())
    {
      std::cout << "Failed to open matrix update for " << matrix_case.name << "\n";
      return error_sum + 1;
    }
    io::updateMatrixFromFile(matrix_update, A.get());
    if (memspace == memory::DEVICE && A->syncData(memory::DEVICE) != 0)
    {
      return error_sum + 1;
    }

    error_sum += solver.setMatrix(A.get());
    if (gcrodr->getActiveRecycleDimension() != retained_recycle_dim)
    {
      std::cout << backend << " " << matrix_case.name
                << ": matrix reset discarded the retained recycle space\n";
      ++error_sum;
    }
    error_sum += solver.resetPreconditioner(A.get());
    error_sum += solveAndCheck(backend,
                               matrix_case.name + std::string("/new-matrix"),
                               solver,
                               matrix_handler,
                               vector_handler,
                               *gcrodr,
                               A.get(),
                               rhs_2.get(),
                               &x,
                               memspace,
                               acceptance_tolerance);

    if (first_recycle_dim == 0 || retained_recycle_dim == 0)
    {
      std::cout << backend << " " << matrix_case.name
                << ": recycle space was not retained between solves\n";
      ++error_sum;
    }

    std::cout << backend << " GCRO-DR " << matrix_case.name
              << (error_sum == 0 ? " PASSED\n" : " FAILED\n");
    return error_sum;
  }
} // namespace

int main(int argc, char* argv[])
{
  ReSolve::CliOptions options(argc, argv);

  auto        option    = options.getParamFromKey("-d");
  std::string data_path = option ? option->second : ".";
  option                = options.getParamFromKey("-c");
  std::string case_name = option ? option->second : "spd";
  option                = options.getParamFromKey("-b");
  std::string backend   = option ? option->second : "cpu";

  const MatrixCase* matrix_case = findMatrixCase(case_name);
  if (matrix_case == nullptr)
  {
    std::cout << "Unknown matrix case " << case_name
              << ". Use spd, symmetric, or asymmetric.\n";
    return 1;
  }

  if (backend == "cpu")
  {
    return runMatrixSequence<ReSolve::LinAlgWorkspaceCpu>(data_path,
                                                          *matrix_case,
                                                          "CPU");
  }

#ifdef RESOLVE_USE_CUDA
  if (backend == "cuda")
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
      std::cout << "No accessible CUDA device; skipping GCRO-DR CUDA test\n";
      return 77;
    }
    return runMatrixSequence<ReSolve::LinAlgWorkspaceCUDA>(data_path,
                                                           *matrix_case,
                                                           "CUDA");
  }
#endif

  std::cout << "Backend " << backend << " is not available in this build\n";
  return 1;
}
