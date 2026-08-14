/**
 * @file ParamTests.hpp
 * @brief Contains definition of ParamTests class.
 * @author Slaven Peles <peless@ornl.org>
 */

#pragma once
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <resolve/GramSchmidt.hpp>
#include <resolve/LinSolverIterativeFGMRES.hpp>
#include <resolve/LinSolverIterativeGCRODR.hpp>
#include <resolve/matrix/Csr.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/VectorHandler.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    /**
     * @brief Class implementing unit tests for Param class.
     *
     * The ParamTests class is implemented entirely in this header file.
     * Adding new unit test requires simply adding another method to this
     * class.
     */
    class ParamTests : TestBase
    {
    public:
      ParamTests()
      {
      }

      virtual ~ParamTests()
      {
      }

      TestOutcome paramSetGet()
      {
        TestStatus success;

        success = true;

        index_type restart   = -1;
        real_type  tol       = -1.0;
        index_type maxit     = -1;
        index_type conv_cond = -1;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();

        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);

        GramSchmidt gs(&vector_handler, GramSchmidt::CGS2);

        // Constructor sets parameters
        LinSolverIterativeFGMRES solver(restart,
                                        tol,
                                        maxit,
                                        conv_cond,
                                        &matrix_handler,
                                        &vector_handler,
                                        &gs);

        // Use getters to read parameters set by the constructor
        index_type restart_out   = solver.getCliParamInt("restart");
        real_type  tol_out       = solver.getCliParamReal("tol");
        index_type maxit_out     = solver.getCliParamInt("maxit");
        index_type conv_cond_out = solver.getCliParamInt("conv_cond");
        bool       flexible_out  = solver.getCliParamBool("flexible");

        // Check getters
        success *= (restart == restart_out);
        success *= (maxit == maxit_out);
        success *= (conv_cond == conv_cond_out);
        success *= isEqual(tol, tol_out);
        success *= flexible_out; // Default is flexible = true

        // Pick different parameter values from the input
        std::string restart_in   = "2";
        std::string tol_in       = "2.0";
        std::string maxit_in     = "2";
        std::string conv_cond_in = "2";
        std::string flexible_in  = "no";

        restart   = atoi(restart_in.c_str());
        tol       = atof(tol_in.c_str());
        maxit     = atoi(maxit_in.c_str());
        conv_cond = atoi(conv_cond_in.c_str());

        // Use setters to change FGMRES solver parameters
        solver.setCliParam("restart", restart_in);
        solver.setCliParam("tol", tol_in);
        solver.setCliParam("maxit", maxit_in);
        solver.setCliParam("conv_cond", conv_cond_in);
        solver.setCliParam("flexible", flexible_in);

        // Read new values
        restart_out   = solver.getCliParamInt("restart");
        tol_out       = solver.getCliParamReal("tol");
        maxit_out     = solver.getCliParamInt("maxit");
        conv_cond_out = solver.getCliParamInt("conv_cond");
        flexible_out  = solver.getCliParamBool("flexible");

        // Check setters
        success *= (restart == restart_out);
        success *= (maxit == maxit_out);
        success *= (conv_cond == conv_cond_out);
        success *= isEqual(tol, tol_out);
        success *= !flexible_out; // flexible was set to "no"

        return success.report(__func__);
      }

      TestOutcome gcrodrParamAndSetup()
      {
        TestStatus success;
        success = true;

        LinAlgWorkspaceCpu workspace;
        workspace.initializeHandles();

        MatrixHandler matrix_handler(&workspace);
        VectorHandler vector_handler(&workspace);
        GramSchmidt   gs(&vector_handler, GramSchmidt::CGS2);

        LinSolverIterativeGCRODR solver(&matrix_handler, &vector_handler, &gs);
        success *= solver.getRestart() == 30;
        success *= solver.getRecycleDimension() == 10;
        success *= solver.getActiveRecycleDimension() == 0;

        success *= solver.setRecycleDimension(4) == 0;
        success *= solver.setRestart(12) == 0;
        success *= solver.setConvergenceCondition(1) == 0;
        success *= solver.setCliParam("tol", "1e-8") == 0;
        success *= solver.setCliParam("maxit", "80") == 0;

        success *= solver.getCliParamInt("restart") == 12;
        success *= solver.getCliParamInt("recycle_dim") == 4;
        success *= solver.getCliParamInt("conv_cond") == 1;
        success *= solver.getCliParamInt("maxit") == 80;
        success *= isEqual(solver.getCliParamReal("tol"), static_cast<real_type>(1e-8));

        success *= solver.setRecycleDimension(12) != 0;
        success *= solver.getRecycleDimension() == 4;
        success *= solver.setRecycleDimension(static_cast<index_type>(-1)) != 0;
        success *= solver.setRestart(4) != 0;
        success *= solver.getRestart() == 12;
        success *= solver.setRestart(static_cast<index_type>(-1)) != 0;
        success *= solver.setConvergenceCondition(3) != 0;
        success *= solver.setConvergenceCondition(static_cast<index_type>(-1)) != 0;
        success *= solver.getConvCond() == 1;
        success *= solver.setCliParam("maxit", "-1") != 0;
        success *= solver.setCliParam("tol", "-1") != 0;

        matrix::Csr A4(4, 4, 4);
        matrix::Csr A7(7, 7, 7);
        matrix::Csr nonsquare(4, 5, 4);
        success *= solver.setup(&A4) == 0;
        success *= solver.setRestart(8) == 0;
        success *= solver.setRecycleDimension(2) == 0;
        success *= solver.resetMatrix(&A4) == 0;
        success *= solver.setup(&A7) == 0;
        success *= solver.setup(&nonsquare) != 0;
        success *= solver.clearRecycleSpace() == 0;
        success *= solver.getActiveRecycleDimension() == 0;

        return success.report(__func__);
      }

    private:
    }; // class ParamTests

  } // namespace tests
} // namespace ReSolve
