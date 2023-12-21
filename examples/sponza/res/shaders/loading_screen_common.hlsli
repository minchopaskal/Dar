#ifndef LOADING_SCREEN_COMMON_HLSLI
#define LOADING_SCREEN_COMMON_HLSLI

#include "interop.hlsli"

struct LoadingScreenConstData {
  UINT width;
  UINT height;
  UINT frame;

  float time;
  float delta;
};

#endif // LOADING_SCREEN_COMMON_HLSLI
