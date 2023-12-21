#pragma once

#include "texture_utils.h"

#include "graphics/frame_data.h"
#include "utils/defines.h"

#include "hud_common.hlsli"

#include <concepts>

struct RectWidgetDesc {
  Optional<String> texture = std::nullopt;
  Optional<Vec4> color = std::nullopt;
  Vec2 topLeft = Vec2{ 0.f, 0.f };
  Vec2 size = Vec2{ 0.f, 0.f };
  uint8_t depth = 0;
};

struct TextWidgetDesc {};

using WidgetHandle = SizeType;
constexpr WidgetHandle INVALID_WIDGET_HANDLE = WidgetHandle(-1);

/// @brief Class rendeing HUD widgets on a texture.
/// @note All positions must be in texture space with top-left(0,0) and bottom-right(1, 1).
/// These coordinates will represent the screen area.
class HUD {
public:
  HUD() = default;

  bool init(Dar::Device &device);

  void beginFrame();
  void endFrame();

  /// @bried Adds text widget.
  /// @note depth \in [0;MAX_DEPTH)
  WidgetHandle addTextWidget(const TextWidgetDesc&) {}

  /// @bried Adds text widget.
  /// @note depth \in [0;MAX_DEPTH)
  WidgetHandle addRectWidget(const RectWidgetDesc& rect);

  WidgetHandle addButton(const RectWidgetDesc& rect);

  bool isOverButton(WidgetHandle handle, Vec2 pos) const;

  WidgetType getWidgetType(WidgetHandle handle);

  /// @bried draw all widgets on screen
  /// @note depth \in [0;MAX_DEPTH)
  bool render();

  /// Get rendered texture
  const Dar::TextureResource& getTexture() const;

  bool resize();

private:
  bool uploadConstData(Dar::UploadHandle handle);
  bool uploadWidgetData(const Vector<WidgetData> &data, Dar::UploadHandle handle);

private:
  static constexpr int MAX_DEPTH = 10;
  static constexpr int MAX_WIDGETS_PER_DEPTH = 256;
  static constexpr int WIDGET_TYPE_OFFSET = MAX_DEPTH * MAX_WIDGETS_PER_DEPTH;

  enum class ConstantBuffers : int {
    ConstData = 0,

    Count
  };

  struct HUDVertex {
    Vec2 position;
    Vec2 texUV;
  };

  using Button = WidgetHandle;

  Vector<TextWidgetDesc> textWidgets[MAX_DEPTH];
  Vector<RectWidgetDesc> rectWidgets[MAX_DEPTH];

  Set<Button> buttons;

  Dar::Renderer hudRenderer;
  Dar::FramePipeline framePipeline;
  Dar::RenderTarget renderTarget;
  Dar::FrameData frameData[Dar::FRAME_COUNT];
  
  Dar::ResourceHandle constData[Dar::FRAME_COUNT];
  Dar::DataBufferResource widgetDataBuffer;
  Dar::VertexBuffer vertexBuffer;
  Dar::IndexBuffer indexBuffer;

  Vector<Dar::TextureResource> textures;
  Vector<TextureDesc> textureDescriptions;
  Dar::HeapHandle texturesHeap;
  Map<String, TextureId> textureName2Id;

  bool validBuffers = false;
};
