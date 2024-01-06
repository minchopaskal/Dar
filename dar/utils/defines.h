#pragma once

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <comdef.h>

#include "dar/utils/logger.h"

// including comdef.h brings these abominations
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#ifdef DAR_DEBUG
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  auto res__ = (cmd); \
  if (!SUCCEEDED(res__)) { \
      LOG_FMT(Error, "Error: %s\n Last Error Code: %lu\n", (msg), res__); \
      char error__[512]; \
      sprintf(error__, "Error: %s. D3D12 Error: %s\n", (msg), _com_error(res__).ErrorMessage()); \
      OutputDebugString(error__); \
      DebugBreak(); \
      return retval; \
  } \
} while (false)
#else
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  auto res__ = (cmd); \
  if (!SUCCEEDED(res__)) { \
      LOG_FMT(Error, "Error: %s\n D3D12 Error(%lu): %s\n", (msg), res__, _com_error(res__).ErrorMessage()); \
      return retval; \
    } \
  } \
while (false)
#endif

#define RETURN_ON_ERROR(cmd, retval, msg) RETURN_ON_ERROR_FMT((cmd), retval, (msg), )

#define RETURN_FALSE_ON_ERROR(cmd, msg) RETURN_ON_ERROR_FMT((cmd), 0, (msg), )

#define RETURN_NULL_ON_ERROR(cmd, msg) RETURN_FALSE_ON_ERROR((cmd), (msg))

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT((cmd), 0, (msg), __VA_ARGS__)

#define RETURN_ERROR_FMT(retval, msg, ...) RETURN_ON_ERROR_FMT(FACILITY_NT_BIT, retval, (msg), __VA_ARGS__)

#define RETURN_ERROR_IF_FMT(cond, retval, msg, ...) \
do { \
if (cond) { \
  RETURN_ERROR_FMT(FACILITY_NT_BIT, retval, (msg), __VA_ARGS__); \
} \
} while (false)

#define RETURN_ERROR(retval, msg) RETURN_ERROR_FMT(FACILITY_NT_BIT, retval, (msg), )

#define RETURN_ERROR_IF(cond, retval, msg) RETURN_ERROR_IF_FMT((cond), retval, (msg), )

#ifdef DAR_DEBUG
#define dassert(exp) \
  if (!(exp)) { \
    DebugBreak(); \
  }

#define dassertLog(exp, msg) \
  if (!(exp)) { \
    LOG_FMT(Error, (msg)); \
    DebugBreak(); \
  }

#else
#define dassert(exp) (void)0
#define dassertLog(exp, msg) (void)0
#endif

#define _DAR_STR(x) #x
#define DAR_STR(x) _DAR_STR(x)
#define TODO(x) static_assert(false, "TODO: " DAR_STR(x) " at " __FILE__ ":" DAR_STR(__LINE__))

using SizeType = size_t;

using D3D12Result = HRESULT;

template <class T>
using Atomic = std::atomic<T>;

using Byte = std::byte;

template <class T>
using Vector = std::vector<T>;

template <class T>
using Queue = std::queue<T>;

template <class T>
using Stack = std::stack<T>;

template <SizeType N>
using Bitset = std::bitset<N>;

template <class K, class V>
using Map = std::unordered_map<K, V>;

template <class T>
using Set = std::unordered_set<T>;

template <class T>
using UniquePtr = std::unique_ptr<T>;

template <class T, SizeType N>
using StaticArray = std::array<T, N>;

using DynamicBitset = std::vector<bool>;

using String = std::string;

using WString = std::wstring;

template <class T>
using Optional = std::optional<T>;

using FenceValue = UINT64;

namespace fs = std::filesystem;
