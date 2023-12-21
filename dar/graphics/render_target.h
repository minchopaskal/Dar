#pragma once

#include "core.h"
#include "d3d12/texture_res.h"
#include "utils/defines.h"
#include <functional>

namespace Dar {

struct RenderTarget {
	std::function<void()> f;
	RenderTarget() : numFramesInFlight(0) {}
	
	void init(TextureInitData &texInitData, UINT numFramesInFlight, const String &name);

	ResourceHandle getHandleForFrame(UINT frameIndex) const;

	ID3D12Resource* getBufferResourceForFrame(UINT frameIndex) const;

	/// Resize the render target buffers with the given dimensions.
	/// Note: Expects the render target to not be in use!
	/// @param width The new width of the render target.
	/// @param height The new height of the render target.
	void resizeRenderTarget(int width, int height);

	void setName(const String &name);

	DXGI_FORMAT getFormat() {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getFormat();
	}

	int getWidth() const {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getWidth();
	}

	int getHeight() const {
		if (numFramesInFlight == 0) {
			return DXGI_FORMAT_UNKNOWN;
		}

		return rtTextures[0].getHeight();
	}

	const TextureResource& getTextureResource(int backbufferIndex) const {
		return rtTextures[backbufferIndex];
	}

private:
	TextureResource rtTextures[FRAME_COUNT];
	UINT numFramesInFlight;
};
} // namespace Dar
