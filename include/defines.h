#pragma once

#include "glm/glm.hpp"

using real = float;

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

//#define RETURN_FALSE_ON_ERROR(cmd, msg) RETURN_ON_ERROR(cmd, false, msg)
static void RETURN_FALSE_ON_ERROR(HRESULT res, const char *msg) {
  if (!SUCCEEDED(res)) {
    auto err = GetLastError();
    OutputDebugString((msg));
    DebugBreak();
  }
}

#define RETURN_FALSE_ON_ERROR_FMT(cmd, msg, ...) RETURN_ON_ERROR_FMT(cmd, false, msg, __VA_ARGS__)

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;

template <int N>
using Vec = glm::vec<N, real>;

using Mat = glm::mat4;
using Mat43 = glm::mat4x3;
