#pragma once

#include <bitset>
#include <cassert>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <comdef.h>

#ifdef D3D12_DEBUG
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  if (!SUCCEEDED((cmd))) { \
      auto err = GetLastError(); \
      fprintf(stderr, "D3D12 Error: %s\n Last Error: %lu\n", (msg), err); \
      char error[512]; sprintf(error, "D3D12 Error: %s\n", _com_error(err).ErrorMessage()); \
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
      fprintf(stderr, "D3D12 Error: %s\n Last error: %lu\n", (msg), (err)); \
      return retval; \
    } \
  } \
while (false)
#endif

#define RETURN_ON_ERROR(cmd, retval, msg) RETURN_ON_ERROR_FMT((cmd), retval, (msg), )

#define RETURN_FALSE_ON_ERROR(cmd, msg) RETURN_ON_ERROR_FMT((cmd), false, (msg), )

#define RETURN_NULL_ON_ERROR(cmd, msg) RETURN_FALSE_ON_ERROR((cmd), (msg))

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT((cmd), false, (msg), __VA_ARGS__)

#ifdef D3D12_DEBUG
#define dassert(exp) assert(exp)
#else
#define dassert(exp) (void)0
#endif

using SizeType = size_t;

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

using String = std::string;
using WString = std::wstring;