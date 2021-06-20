#pragma once

#include <vector>
#include <queue>
#include <bitset>
#include <string>

#ifdef D3D12_DEBUG
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  if (!SUCCEEDED((cmd))) { \
      auto err = GetLastError(); \
      fprintf(stderr, "D3D12 Error: %s\n", (msg)); \
      char error[512]; sprintf(error, "D3D12 Error: %lu\n", err); \
      OutputDebugString(error); \
      DebugBreak(); \
      return retval; \
  } \
} while (false)
#else
#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
do { \
  if (!SUCCEEDED((res))) { \
      auto err = GetLastError(); \
      fprintf(stderr, "D3D12 Error: %s. Last error: %lu\n", (msg), (err)); \
      return (retval); \
    } \
  } \
while (false)
#endif

#define RETURN_ON_ERROR(cmd, retval, msg) RETURN_ON_ERROR_FMT((cmd), retval, (msg), )

#define RETURN_FALSE_ON_ERROR(cmd, msg) RETURN_ON_ERROR_FMT((cmd), false, (msg), )

#define RETURN_NULL_ON_ERROR(cmd, msg) RETURN_FALSE_ON_ERROR((cmd), (msg))

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT((cmd), false, (msg), __VA_ARGS__)

template <class T>
using Vector = std::vector<T>;

template <class T>
using Queue = std::queue<T>;

template <size_t N>
using Bitset = std::bitset<N>;

using String = std::string;
using WString = std::wstring;