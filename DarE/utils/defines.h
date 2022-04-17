#pragma once

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <comdef.h>

#include "utils/logger.h"

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
  if (!SUCCEEDED((cmd))) { \
      auto err = GetLastError(); \
      LOG_FMT(Error, "D3D12 Error: %s\n Last Error: %lu\n", (msg), err); \
      char error[512]; \
      sprintf(error, "D3D12 Error: %s\n", _com_error(err).ErrorMessage()); \
      OutputDebugString(error); \
      DebugBreak(); \
      return retval; \
  } \
} while (false)
#else
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  if (!SUCCEEDED(cmd)) { \
      auto err = GetLastError(); \
      LOG_FMT(Error, "D3D12 Error: %s\n Last Error: %lu\n", (msg), err); \
      return retval; \
    } \
  } \
while (false)
#endif

#define RETURN_ON_ERROR(cmd, retval, msg) RETURN_ON_ERROR_FMT((cmd), retval, (msg), )

#define RETURN_FALSE_ON_ERROR(cmd, msg) RETURN_ON_ERROR_FMT((cmd), false, (msg), )

#define RETURN_NULL_ON_ERROR(cmd, msg) RETURN_FALSE_ON_ERROR((cmd), (msg))

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT((cmd), false, (msg), __VA_ARGS__)

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
#else
#define dassert(exp) (void)0
#endif

using SizeType = size_t;

template <class T>
using Atomic = std::atomic<T>;

using Byte = std::byte;

template <class T>
using Vector = std::vector<T>;

template <class T>
using Queue = std::queue<T>;

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