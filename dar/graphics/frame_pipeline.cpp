#include "graphics/frame_pipeline.h"

namespace Dar {

FramePipeline::FramePipeline() {}

void FramePipeline::deinit() {
	for (int i = 0; i < renderPasses.size(); ++i) {
		renderPasses[i]->deinit();
	}
	renderPasses.clear();
	renderPassesStorage.clear();
}

void FramePipeline::addRenderPass(const RenderPassDesc &rpd) {
	renderPassesDesc.push_back(rpd);
}

void FramePipeline::compilePipeline(Device &device) {
	deinit();

	const SizeType numRenderPasses = renderPassesDesc.size();
	renderPassesStorage.resize(numRenderPasses * sizeof(RenderPass));
	for (SizeType i = 0; i < numRenderPasses; ++i) {
		RenderPass *renderPass = new (renderPassesStorage.data() + i * sizeof(RenderPass)) RenderPass;
		renderPass->init(device.getDevice(), &device.getBackbuffer(), renderPassesDesc[i]);

		renderPasses.push_back(renderPass);
	}

	// We no longer need the memory since the render passes are initialized.
	renderPassesDesc.clear();
}

} // namespace Dar
