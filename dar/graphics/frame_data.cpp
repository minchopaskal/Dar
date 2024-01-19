#pragma once

#include "frame_data.h"

namespace Dar {

void FrameData::clear() {
	useSameCommands = false;
	vertexBuffers.clear();
	indexBuffers.clear();
	constantBuffers.clear();
	shaderResources.clear();
	renderCommands.clear();
	uploadsToWait.clear();
	fencesToWait.clear();
	clearRenderTargets.clear();
}

void FrameData::beginFrame(const Renderer &renderer) {
	passIndex = -1;
	constantBuffers.clear();
	shaderResources.clear();
	uploadsToWait.clear();
	fencesToWait.clear();

	clearRenderTargets.resize(renderer.getNumPasses());
	std::fill(clearRenderTargets.begin(), clearRenderTargets.end(), true);

	shaderResources.resize(renderer.getNumPasses());

	vertexBuffers.resize(renderer.getNumPasses());
	std::fill(vertexBuffers.begin(), vertexBuffers.end(), nullptr);
	indexBuffers.resize(renderer.getNumPasses());
	std::fill(indexBuffers.begin(), indexBuffers.end(), nullptr);
	
	if (!useSameCommands) {
		renderCommands.clear();
		renderCommands.resize(renderer.getNumPasses());
	}
}

void FrameData::startNewPass() {
	dassert(passIndex + 1 < renderCommands.size());
	++passIndex;
}

} // namespace Dar