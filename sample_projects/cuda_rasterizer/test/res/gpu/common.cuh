#ifndef COMMON_CUH
#define COMMON_CUH

#if defined(__CUDACC__)
#define ALIGNAS(x) __align__((x))
#elif defined(_MSC_VER)
#define ALIGNAS(x) __declspec(align((x)))
#else
static_assert(false, "Provide ALIGNAS macro for your compiler!")
#endif // __CUDACC__

#endif // COMMON_CUH
