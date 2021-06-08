#pragma once

#include <vector>
#include <queue>
#include <bitset>
#include <string>

#define RETURN_ON_ERROR(cmd, retval, msg) \
if (!SUCCEEDED(cmd)) { \
  OutputDebugString((msg)); \
  DebugBreak(); \
  return retval; \
}

#define RETURN_ON_ERROR_FMT(cmd, retval, msg, ...) \
if (!SUCCEEDED(cmd)) { \
  fprintf(stderr, (msg), __VA_ARGS__); \
  return retval; \
}

//static void RETURN_FALSE_ON_ERROR(HRESULT res, const char *msg) {
#define RETURN_FALSE_ON_ERROR(res, msg) \
do { \
  if (!SUCCEEDED(res)) { \
      auto err = GetLastError(); \
      OutputDebugString((msg)); \
      DebugBreak(); \
      return false; \
    } \
  } \
while (false)

#define RETURN_NULL_ON_ERROR(res, msg) RETURN_FALSE_ON_ERROR((res), (msg))

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT(cmd, false, msg, __VA_ARGS__)

template <class T>
using Vector = std::vector<T>;

template <class T>
using Queue = std::queue<T>;

template <size_t N>
using Bitset = std::bitset<N>;

using String = std::string;
using WString = std::wstring;