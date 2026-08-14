/**
 * @file gcrodrReuseBenchmark.cpp
 * @brief Compare FGMRES and GCRO-DR with and without cross-solve reuse.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/LinSolverIterativeGCRODR.hpp>
#include <resolve/Profiling.hpp>
#include <resolve/SystemSolver.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/matrix/io.hpp>
#include <resolve/utilities/params/CliOptions.hpp>
#include <resolve/vector/Vector.hpp>
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
  using Clock      = std::chrono::steady_clock;

  struct SolveResult
  {
    int        status{0};
    index_type iterations{0};
    index_type recycle_dimension{0};
    real_type  relative_residual{0.0};
    double     seconds{0.0};
  };

  void printHelp()
  {
    std::cout << "Usage: gcrodrReuseBenchmark.exe [options]\n"
              << "  -b <cpu|cuda>  Backend (default: cpu)\n"
              << "  -d <path>      ReSolve functionality data directory\n"
              << "  -m <restart>   GCRO-DR restart dimension (default: 40)\n"
              << "  -k <recycle>   Recycle target dimension (default: 20)\n"
              << "  -n <maxit>     Maximum iterations (default: 5000)\n"
              << "  -t <tol>       Relative residual tolerance (default: 1e-10)\n";
  }

  std::unique_ptr<MatrixType> loadMatrix(const std::string& path)
  {
    std::ifstream input(path);
    if (!input.is_open())
    {
      std::cout << "Failed to open matrix file " << path << "\n";
      return std::unique_ptr<MatrixType>();
    }
    return std::unique_ptr<MatrixType>(
        ReSolve::io::createCsrFromFile(input, true));
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
                        VectorType*                  rhs1,
                        VectorType*                  rhs2,
                        ReSolve::memory::MemorySpace memspace)
  {
    if (memspace == ReSolve::memory::HOST)
    {
      return 0;
    }

    int status = A->allocateMatrixData(memspace);
    status += A->syncData(memspace);
    status += rhs1->allocate(memspace);
    status += rhs1->syncData(memspace);
    status += rhs2->allocate(memspace);
    status += rhs2->syncData(memspace);
    return status;
  }

  SolveResult timedSolve(ReSolve::SystemSolver&       solver,
                         ReSolve::LinSolverIterative& iterative_solver,
                         VectorType*                  rhs,
                         VectorType*                  x,
                         ReSolve::memory::MemorySpace memspace,
                         const char*                  profile_label)
  {
    x->setToZero(memspace);
    Clock::time_point start;
    Clock::time_point end;
    int               status = 0;
    {
      RESOLVE_PROFILE_SCOPE(profile_label);
      start  = Clock::now();
      status = solver.solve(rhs, x);
      end    = Clock::now();
    }

    SolveResult result;
    result.status            = status;
    result.seconds           = std::chrono::duration<double>(end - start).count();
    result.iterations        = iterative_solver.getNumIter();
    const auto* gcrodr = dynamic_cast<const ReSolve::LinSolverIterativeGCRODR*>(
        &iterative_solver);
    result.recycle_dimension =
        gcrodr == nullptr ? 0 : gcrodr->getActiveRecycleDimension();
    if (status == 0)
    {
      result.relative_residual = solver.getResidualNorm(rhs, x);
    }
    else
    {
      result.relative_residual =
          std::numeric_limits<real_type>::infinity();
    }
    return result;
  }

  void printResult(const std::string& label, const SolveResult& result)
  {
    std::cout << std::left << std::setw(16) << label
              << " iterations=" << std::right << std::setw(5)
              << result.iterations
              << "  solve_seconds=" << std::fixed << std::setprecision(6)
              << result.seconds
              << "  relative_residual=" << std::scientific
              << std::setprecision(6) << result.relative_residual
              << "  active_recycle_dim=" << result.recycle_dimension << "\n";
  }

  template <class WorkspaceType>
  int runBenchmark(const std::string& data_path,
                   const std::string& backend,
                   index_type         restart,
                   index_type         recycle_dimension,
                   index_type         max_iterations,
                   real_type          tolerance)
  {
    using namespace ReSolve;

    WorkspaceType workspace;
    workspace.initializeHandles();
    MatrixHandler             matrix_handler(&workspace);
    const memory::MemorySpace memspace =
        matrix_handler.getIsCudaEnabled() ? memory::DEVICE : memory::HOST;

    const std::string rhs_prefix =
        data_path + "/rhs_ACTIVSg200_AC_renumbered_add9_ones_";
    std::unique_ptr<MatrixType> A = loadMatrix(
        data_path + "/matrix_ACTIVSg200_AC_renumbered_add9_01.mtx");
    // The second sequence RHS is deliberately used with the same matrix. It
    // provides an independent RHS while keeping every non-recycling input
    // identical between the reused and cold measurements.
    std::unique_ptr<VectorType> rhs1 = loadVector(rhs_prefix + "01.mtx");
    std::unique_ptr<VectorType> rhs2 = loadVector(rhs_prefix + "02.mtx");
    if (!A || !rhs1 || !rhs2 || A->getNumRows() != rhs1->getSize()
        || A->getNumRows() != rhs2->getSize())
    {
      std::cout << "Invalid ACTIVSg200 benchmark inputs\n";
      return 1;
    }
    if (moveToMemorySpace(A.get(),
                          rhs1.get(),
                          rhs2.get(),
                          memspace)
        != 0)
    {
      std::cout << "Failed to prepare " << backend << " input data\n";
      return 1;
    }

    SystemSolver solver(&workspace, "none", "none", "gcrodr", "ilu0", "none");
    auto*        gcrodr = dynamic_cast<LinSolverIterativeGCRODR*>(
        &solver.getIterativeSolver());
    if (gcrodr == nullptr)
    {
      return 1;
    }
    int status = gcrodr->setRestart(restart);
    status += gcrodr->setRecycleDimension(recycle_dimension);
    gcrodr->setMaxit(max_iterations);
    gcrodr->setTol(tolerance);
    status += solver.setMatrix(A.get());
    status += solver.preconditionerSetup("right");
    if (status != 0)
    {
      std::cout << "Failed to configure the benchmark solver\n";
      return status;
    }

    VectorType x(A->getNumRows());
    x.allocate(memspace);

    // Build a recycle space on the first system. This cost is part of solving
    // the sequence regardless of how the second system is handled.
    const SolveResult training =
        timedSolve(solver,
                   *gcrodr,
                   rhs1.get(),
                   &x,
                   memspace,
                   "solve.GCRODR.train");
    if (training.status != 0 || training.recycle_dimension == 0)
    {
      std::cout << "Failed to build a recycle space on the first system\n";
      return 1;
    }

    // Keep A and its preconditioner fixed so the comparison isolates the
    // benefit of retaining the GCRO-DR space between different right-hand
    // sides. Since neither operator changed, the reused solve can use the
    // existing image of the recycle space without refreshing it.
    const SolveResult reused =
        timedSolve(solver,
                   *gcrodr,
                   rhs2.get(),
                   &x,
                   memspace,
                   "solve.GCRODR.reuse");

    // Clear only the recycle space and solve the identical A/rhs2 system
    // again. Running cold second gives it the favorable cache order.
    gcrodr->clearRecycleSpace();
    const SolveResult cold =
        timedSolve(solver,
                   *gcrodr,
                   rhs2.get(),
                   &x,
                   memspace,
                   "solve.GCRODR.cold");

    // FGMRES is the practical no-recycling baseline. Give it an independent
    // solver and ILU0 instance, but use the same matrix, right-preconditioning,
    // restart, CGS2 orthogonalization, tolerance, and zero initial guesses.
    SystemSolver fgmres_solver(
        &workspace, "none", "none", "fgmres", "ilu0", "none");
    auto* fgmres = dynamic_cast<LinSolverIterativeFGMRES*>(
        &fgmres_solver.getIterativeSolver());
    if (fgmres == nullptr)
    {
      return 1;
    }
    status = fgmres->setRestart(restart);
    status += fgmres->setFlexible(true);
    fgmres->setMaxit(max_iterations);
    fgmres->setTol(tolerance);
    status += fgmres_solver.setMatrix(A.get());
    status += fgmres_solver.preconditionerSetup("right");
    if (status != 0)
    {
      std::cout << "Failed to configure the FGMRES baseline\n";
      return status;
    }
    const SolveResult fgmres_first =
        timedSolve(fgmres_solver,
                   *fgmres,
                   rhs1.get(),
                   &x,
                   memspace,
                   "solve.FGMRES.b1");
    const SolveResult fgmres_second =
        timedSolve(fgmres_solver,
                   *fgmres,
                   rhs2.get(),
                   &x,
                   memspace,
                   "solve.FGMRES.b2");

    std::cout << "\nFGMRES/GCRO-DR recycle benchmark on " << backend << "\n"
              << "sequence=ACTIVSg200_A1/b1->A1/b2"
              << "  restart=" << restart
              << "  recycle_target=" << recycle_dimension
              << "  tolerance=" << std::scientific << tolerance << "\n";
    printResult("GCRODR train", training);
    printResult("GCRODR reuse", reused);
    printResult("GCRODR cold", cold);
    printResult("FGMRES b1", fgmres_first);
    printResult("FGMRES b2", fgmres_second);

    const double iteration_reduction =
        cold.iterations == 0
            ? 0.0
            : 100.0
                  * (1.0 - static_cast<double>(reused.iterations)
                               / static_cast<double>(cold.iterations));
    const double reuse_vs_cold_speedup =
        reused.seconds == 0.0 ? 0.0 : cold.seconds / reused.seconds;
    const double reuse_vs_cold_sequence_speedup =
        training.seconds + reused.seconds == 0.0
            ? 0.0
            : (training.seconds + cold.seconds)
                  / (training.seconds + reused.seconds);
    const double reuse_vs_fgmres_iteration_reduction =
        fgmres_second.iterations == 0
            ? 0.0
            : 100.0
                  * (1.0 - static_cast<double>(reused.iterations)
                               / static_cast<double>(fgmres_second.iterations));
    const double fgmres_vs_cold_speedup =
        fgmres_second.seconds == 0.0
            ? 0.0
            : cold.seconds / fgmres_second.seconds;
    const double fgmres_vs_reuse_speedup =
        fgmres_second.seconds == 0.0
            ? 0.0
            : reused.seconds / fgmres_second.seconds;
    const double fgmres_vs_reuse_sequence_speedup =
        fgmres_first.seconds + fgmres_second.seconds == 0.0
            ? 0.0
            : (training.seconds + reused.seconds)
                  / (fgmres_first.seconds + fgmres_second.seconds);
    std::cout << std::fixed << std::setprecision(2)
              << "reuse_vs_cold_iteration_reduction_percent="
              << iteration_reduction << "\n"
              << "reuse_vs_cold_second_solve_speedup="
              << reuse_vs_cold_speedup << "x\n"
              << "reuse_vs_cold_two_rhs_speedup="
              << reuse_vs_cold_sequence_speedup << "x\n"
              << "reuse_vs_fgmres_iteration_reduction_percent="
              << reuse_vs_fgmres_iteration_reduction << "\n"
              << "fgmres_vs_gcrodr_cold_speedup="
              << fgmres_vs_cold_speedup << "x\n"
              << "fgmres_vs_gcrodr_reuse_speedup="
              << fgmres_vs_reuse_speedup << "x\n"
              << "fgmres_vs_gcrodr_reuse_two_rhs_speedup="
              << fgmres_vs_reuse_sequence_speedup << "x\n";

    const real_type residual_limit =
        std::max(static_cast<real_type>(100.0) * tolerance,
                 static_cast<real_type>(1.0e-8));
    if (reused.status != 0 || cold.status != 0 || fgmres_first.status != 0
        || fgmres_second.status != 0
        || !std::isfinite(reused.relative_residual)
        || !std::isfinite(cold.relative_residual)
        || !std::isfinite(fgmres_first.relative_residual)
        || !std::isfinite(fgmres_second.relative_residual)
        || reused.relative_residual > residual_limit
        || cold.relative_residual > residual_limit
        || fgmres_first.relative_residual > residual_limit
        || fgmres_second.relative_residual > residual_limit)
    {
      return 1;
    }
    return 0;
  }
} // namespace

int main(int argc, char* argv[])
{
  ReSolve::CliOptions options(argc, argv);
  if (options.getParamFromKey("-h"))
  {
    printHelp();
    return 0;
  }

  auto        option       = options.getParamFromKey("-b");
  std::string backend      = option ? option->second : "cpu";
  option                   = options.getParamFromKey("-d");
  std::string data_path    = option ? option->second : "tests/functionality/data";
  option                   = options.getParamFromKey("-m");
  const index_type restart = option
                                 ? static_cast<index_type>(std::strtoull(
                                       option->second.c_str(), nullptr, 10))
                                 : static_cast<index_type>(40);
  option                   = options.getParamFromKey("-k");
  const index_type recycle_dimension =
      option ? static_cast<index_type>(
                   std::strtoull(option->second.c_str(), nullptr, 10))
             : static_cast<index_type>(20);
  option = options.getParamFromKey("-n");
  const index_type max_iterations =
      option ? static_cast<index_type>(
                   std::strtoull(option->second.c_str(), nullptr, 10))
             : static_cast<index_type>(5000);
  option                    = options.getParamFromKey("-t");
  const real_type tolerance = option
                                  ? static_cast<real_type>(
                                        std::strtod(option->second.c_str(), nullptr))
                                  : static_cast<real_type>(1.0e-10);

  if (backend == "cpu")
  {
    return runBenchmark<ReSolve::LinAlgWorkspaceCpu>(data_path,
                                                     "CPU",
                                                     restart,
                                                     recycle_dimension,
                                                     max_iterations,
                                                     tolerance);
  }

#ifdef RESOLVE_USE_CUDA
  if (backend == "cuda")
  {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
      std::cout << "No accessible CUDA device\n";
      return 77;
    }
    return runBenchmark<ReSolve::LinAlgWorkspaceCUDA>(data_path,
                                                      "CUDA",
                                                      restart,
                                                      recycle_dimension,
                                                      max_iterations,
                                                      tolerance);
  }
#endif

  std::cout << "Backend " << backend << " is unavailable in this build\n";
  return 1;
}
