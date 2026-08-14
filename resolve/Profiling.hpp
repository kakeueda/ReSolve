#pragma once

#ifdef RESOLVE_USE_PROFILING

#ifdef RESOLVE_USE_GPU
#ifdef RESOLVE_USE_HIP
#include <rocprofiler-sdk-roctx/roctx.h>
#define RESOLVE_RANGE_PUSH(x) roctxRangePush(x)
#define RESOLVE_RANGE_POP(x) \
  roctxRangePop();           \
  roctxMarkA(x)
#endif // RESOLVE_USE_HIP

#ifdef RESOLVE_USE_CUDA
#include <nvtx3/nvToolsExt.h>
#define RESOLVE_RANGE_PUSH(x) nvtxRangePush(x)
#define RESOLVE_RANGE_POP(x) \
  nvtxRangePop();            \
  nvtxMarkA(x)
#endif // RESOLVE_USE_CUDA

#else

// Not using GPU
#define RESOLVE_RANGE_PUSH(x)
#define RESOLVE_RANGE_POP(x)

#endif // RESOLVE_USE_GPU

#else

// Not using profiling
#define RESOLVE_RANGE_PUSH(x)
#define RESOLVE_RANGE_POP(x)

#endif // RESOLVE_USE_PROFILING

namespace ReSolve
{
  class ProfilingRange
  {
  public:
    explicit ProfilingRange(const char* name)
      : name_(name)
    {
      RESOLVE_RANGE_PUSH(name_);
    }

    ~ProfilingRange()
    {
      RESOLVE_RANGE_POP(name_);
    }

    ProfilingRange(const ProfilingRange&)            = delete;
    ProfilingRange& operator=(const ProfilingRange&) = delete;

  private:
    const char* name_;
  };
} // namespace ReSolve

#define RESOLVE_PROFILE_CONCAT_INNER(x, y) x##y
#define RESOLVE_PROFILE_CONCAT(x, y) RESOLVE_PROFILE_CONCAT_INNER(x, y)
#define RESOLVE_PROFILE_SCOPE(name)                                         \
  ::ReSolve::ProfilingRange RESOLVE_PROFILE_CONCAT(resolve_profile_range_, \
                                                    __LINE__)(name)
