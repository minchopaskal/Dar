#pragma once

#include "device.h"
#include "render_pass.h"
#include "utils/defines.h"

namespace Dar {

class FramePipeline {
public:
	FramePipeline();
	~FramePipeline() {
		deinit();
	}

	void init() {}
	void deinit();

	void addRenderPass(const RenderPassDesc &rpd);

	/// Pipeline used in a different meaning than a graphics API pipeline here.
	/// The renderer's pipeline comprises of all render passes and the order in which they apper.
	/// Compiles the render passes and is ready to be used by a Renderer instance.
	/// Any subsequent calls to addRenderPass will start creating a new frame pipeline that
	/// needs to be compiled again. Subsequent calls to compilePipeline deinitialize the current state
	/// and compile only the render passes that were added after the previous call to compilePipeline.
	void compilePipeline(Device &device);

	RenderPass &getPass(SizeType index) {
		dassert(renderPasses.size() > index && renderPasses[index]);

		return *renderPasses[index];
	}

	SizeType getNumPasses() const {
		return renderPasses.size();
	}

private:
	Vector<RenderPassDesc> renderPassesDesc;
	Vector<RenderPass*> renderPasses;
	Vector<Byte> renderPassesStorage;
};

} // namespace Da
