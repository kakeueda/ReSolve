#pragma once
#include <iomanip>
#include <string>
#include <vector>

#include <resolve/GramSchmidt.hpp>
#include <resolve/matrix/MatrixHandler.hpp>
#include <resolve/vector/Vector.hpp>
#include <resolve/workspace/LinAlgWorkspace.hpp>
#include <tests/unit/TestBase.hpp>

namespace ReSolve
{
  namespace tests
  {
    const real_type var1 = 0.17;
    const real_type var2 = 2.0;

    class GramSchmidtTests : TestBase
    {
    public:
      GramSchmidtTests(ReSolve::VectorHandler& handler)
        : handler_(handler)
      {
        if (handler_.getIsCudaEnabled() || handler_.getIsHipEnabled())
          memspace_ = memory::DEVICE;
        else
          memspace_ = memory::HOST;
      }

      virtual ~GramSchmidtTests()
      {
      }

      TestOutcome GramSchmidtConstructor()
      {
        TestStatus status;

        VectorHandler vh;
        GramSchmidt   gs2(&vh, GramSchmidt::MGS_PM);
        status *= (gs2.getVariant() == GramSchmidt::MGS_PM);

        return status.report(__func__);
      }

      TestOutcome orthogonalize(index_type N, GramSchmidt::GSVariant var)
      {
        TestStatus status;

        // Set test name
        std::string testname(__func__);
        switch (var)
        {
        case GramSchmidt::MGS:
          testname += " (Modified Gram-Schmidt)";
          break;
        case GramSchmidt::MGS_TWO_SYNC:
          testname += " (Modified Gram-Schmidt 2-Sync)";
          break;
        case GramSchmidt::MGS_PM:
          testname += " (Post-Modern Modified Gram-Schmidt)";
          break;
        case GramSchmidt::CGS1:
          testname += " (Classical Gram-Schmidt)";
          break;
        case GramSchmidt::CGS2:
          testname += " (Reorthogonalized Classical Gram-Schmidt)";
          break;
        }

        // Answer key designed for restart = 2
        index_type restart = 2;

        // Krylov space spanned by 3 vectors
        vector::Vector V(N, restart + 1);

        // Hessenberg matrix size is 2 x 3
        real_type* H = new real_type[restart * (restart + 1)];

        // Allocate Krylov subspace
        V.allocateAll(memspace_);
        ;
        V.setDataUpdated(memspace_);
        if (memspace_ == memory::DEVICE)
        {
          V.setDataUpdated(memory::HOST);
        }

        // Create and allocate Gram-Schmidt orthogonalization
        ReSolve::GramSchmidt GS(&handler_, var);
        GS.setup(N, restart);

        // Fill 2nd and 3rd vector with values
        real_type* aux_data = V.getData(1, memory::HOST);
        for (int i = 0; i < N; ++i)
        {
          if (i % 2 == 0)
          {
            aux_data[i] = constants::ONE;
          }
          else
          {
            aux_data[i] = var1;
          }
        }
        aux_data = V.getData(2, memory::HOST);
        for (int i = 0; i < N; ++i)
        {
          if (i % 3 > 0)
          {
            aux_data[i] = constants::ZERO;
          }
          else
          {
            aux_data[i] = var2;
          }
        }
        V.setDataUpdated(memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          V.syncData(memspace_);
        }

        // Set the first vector to all 1s and normalize it.
        V.setToConst(0, 1.0, memspace_);
        real_type nrm = handler_.dot(&V, &V, memspace_);
        nrm           = sqrt(nrm);
        nrm           = 1.0 / nrm;
        handler_.scal(nrm, &V, memspace_);

        // Orthogonalize system and verify result
        if (memspace_ == memory::DEVICE)
        {
          V.syncData(memory::HOST);
        }
        GS.orthogonalize(N, &V, H, 0);
        GS.orthogonalize(N, &V, H, 1);
        status *= verifyAnswer(V, restart + 1);

        delete[] H;

        return status.report(testname.c_str());
      }

      TestOutcome arnoldiRelation(index_type N, GramSchmidt::GSVariant var)
      {
        TestStatus status;

        const index_type restart = 1;
        vector::Vector   V(N, restart + 1);
        V.allocateAll(memspace_);

        // Normalize the first Arnoldi basis vector.
        V.setToConst(0, constants::ONE, memspace_);
        real_type norm = handler_.dot(&V, &V, memspace_);
        handler_.scal(constants::ONE / std::sqrt(norm), &V, memspace_);

        // Store a nontrivial candidate w in V(:,1) and retain it so the
        // Arnoldi relation w = h(0,0) v_0 + h(1,0) v_1 can be checked.
        std::vector<real_type> candidate(static_cast<std::size_t>(N));
        real_type*             working = V.getData(1, memory::HOST);
        for (index_type row = 0; row < N; ++row)
        {
          const real_type value =
              static_cast<real_type>(row % 7) - static_cast<real_type>(3)
              + static_cast<real_type>(0.125 * (row % 3));
          candidate[static_cast<std::size_t>(row)] = value;
          working[row]                             = value;
        }
        V.setDataUpdated(1, memory::HOST);
        if (memspace_ == memory::DEVICE)
        {
          V.syncData(1, memory::DEVICE);
        }

        GramSchmidt GS(&handler_, var);
        GS.setup(N, restart);
        real_type H[2] = {constants::ZERO, constants::ZERO};
        status *= (GS.orthogonalize(N, &V, H, 0) == 0);

        std::vector<real_type> v0(static_cast<std::size_t>(N));
        std::vector<real_type> v1(static_cast<std::size_t>(N));
        status *= (V.copyToExternal(v0.data(), 0, memspace_, memory::HOST) == 0);
        status *= (V.copyToExternal(v1.data(), 1, memspace_, memory::HOST) == 0);
        real_type        error_squared     = constants::ZERO;
        real_type        candidate_squared = constants::ZERO;
        for (index_type row = 0; row < N; ++row)
        {
          const real_type reconstructed =
              H[0] * v0[static_cast<std::size_t>(row)]
              + H[1] * v1[static_cast<std::size_t>(row)];
          const real_type difference =
              candidate[static_cast<std::size_t>(row)] - reconstructed;
          error_squared += difference * difference;
          candidate_squared += candidate[static_cast<std::size_t>(row)]
                               * candidate[static_cast<std::size_t>(row)];
        }
        const real_type relative_error =
            std::sqrt(error_squared / candidate_squared);
        const real_type tolerance =
            static_cast<real_type>(100 * N)
            * std::numeric_limits<real_type>::epsilon();
        if (relative_error > tolerance)
        {
          std::cout << "Arnoldi relation relative error: " << relative_error
                    << ", tolerance: " << tolerance << "\n";
          status *= false;
        }

        return status.report(__func__);
      }

    private:
      ReSolve::VectorHandler&      handler_;
      ReSolve::memory::MemorySpace memspace_;

      // x is a multivector containing K vectors
      bool verifyAnswer(vector::Vector& x, index_type K)
      {
        vector::Vector a(x.getSize());
        a.allocate(memory::HOST);
        vector::Vector b(x.getSize());
        b.allocate(memory::HOST);

        real_type ip;
        bool      status = true;

        for (index_type i = 0; i < K; ++i)
        {
          for (index_type j = 0; j < K; ++j)
          {
            a.copyFromExternal(x.getData(i, memspace_), memspace_, memory::HOST);
            b.copyFromExternal(x.getData(j, memspace_), memspace_, memory::HOST);
            ip = handler_.dot(&a, &b, memory::HOST);
            if ((i != j) && !isEqual(ip, 0.0))
            {
              status = false;
              std::cout << "Vectors " << i << " and " << j << " are not orthogonal!"
                        << " Inner product computed: " << ip << ", expected: " << 0.0 << "\n";
              break;
            }
            if ((i == j) && !isEqual(sqrt(ip), 1.0))
            {
              status = false;
              std::cout << std::setprecision(16);
              std::cout << "Vector " << i << " has norm: " << sqrt(ip)
                        << " expected: " << 1.0 << "\n";
              break;
            }
          }
        }

        return status;
      }
    }; // class
  } // namespace tests
} // namespace ReSolve
