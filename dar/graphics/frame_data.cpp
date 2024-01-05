#pragma once

#include "frame_data.h"

namespace Dar {

void FrameData::clear() {
	useSameCommands = false;
	vertexBuffer = nullptr;
	indexBuffer = nullptr;
	constantBuffers.clear();
	shaderResources.clear();
	renderCommands.clear();
	uploadsToWait.clear();
	fencesToWait.clear();
}

void FrameData::beginFrame(const Renderer &renderer) {
	passIndex = -1;
	vertexBuffer = nullptr;
	indexBuffer = nullptr;
	constantBuffers.clear();
	shaderResources.clear();
	uploadsToWait.clear();
	fencesToWait.clear();
	shaderResources.resize(renderer.getNumPasses());
	
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