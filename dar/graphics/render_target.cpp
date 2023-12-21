#pragma once

#include "render_target.h"

namespace Dar {

void RenderTarget::init(TextureInitData &texInitData, UINT numFrames, const String &name) {
	dassert(numFrames <= FRAME_COUNT);
	numFrames = std::min(numFrames, FRAME_COUNT);
	numFramesInFlight = numFrames;

	for (UINT i = 0; i < numFramesInFlight; ++i) {
		rtTextures[i].init(texInitData, TextureResourceType::RenderTarget, name);
	}
}

ResourceHandle RenderTarget::getHandleForFrame(UINT frameIndex) const {
	dassert(frameIndex < numFramesInFlight);
	if (frameIndex >= numFramesInFlight) {
		return INVALID_RESOURCE_HANDLE;
	}

	return rtTextures[frameIndex].getHandle();
}

ID3D12Resource *RenderTarget::getBufferResourceForFrame(UINT frameIndex) const {
	dassert(frameIndex < numFramesInFlight);
	if (frameIndex >= numFramesInFlight) {
		return nullptr;
	}

	return rtTextures[frameIndex].getBufferResource();
}

void RenderTarget::resizeRenderTarget(int width, int height) {
	Dar::TextureInitData rtvTextureDesc = {};
	rtvTextureDesc.width = width;
	rtvTextureDesc.height = height;
	rtvTextureDesc.format = getFormat();
	rtvTextureDesc.clearValue.color[0] = 0.f;
	rtvTextureDesc.clearValue.color[1] = 0.f;
	rtvTextureDesc.clearValue.color[2] = 0.f;
	rtvTextureDesc.clearValue.color[3] = 0.f;

	for (UINT i = 0; i < numFramesInFlight; ++i) {
		TextureResource &t = rtTextures[i];
		const String &name = t.getName();

		t.init(rtvTextureDesc, TextureResourceType::RenderTarget, name);
	}
}

void RenderTarget::setName(const String &name) {
	for (UINT i = 0; i < numFramesInFlight; ++i) {
		rtTextures[i].setName(name + "[" + std::to_string(i) + "]");
	}
}

} // namespace Dar