#ifndef HUD_COMMON_HLSLI
#define HUD_COMMON_HLSLI

#include "interop.hlsli"

enum WidgetType {
  Text = 0,
  Rect,

  WidgetCount,
  InvalidWidget,
};

struct WidgetData {
  Vec4 color;
  int textureIndex;
  WidgetType type;
};

struct HUDConstData {
  Mat4 projection;
  float width;
  float height;
};

#endif // HUD_COMMON_HLSLI
