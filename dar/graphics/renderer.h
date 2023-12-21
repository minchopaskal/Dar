#pragma once

#include "backbuffer.h"
#include "device.h"
#include "frame_data.h"
#include "frame_pipeline.h"
#include "render_pass.h"
#include "math/dar_math.h"
#include "utils/defines.h"

namespace Dar {

struct RenderSettings {
	int showGBuffer = 0;
	bool enableNormalMapping= 1;
	bool enableFXAA = 0;
	bool useImGui = 0; ///< Flag indicating whether ImGui will be used for drawing UI.
	bool vSyncEnabled = 0;
	bool spotLightON = 0;
};

class Renderer {
public:
	Renderer();
	
	/// Initialize the renderer
	/// @param device Device used for rendering
	/// @param renderToScreen Flag indicating the renderer will output to the backbuffer
	/// @note It's forbidden to have more than one renderer rendering to the backbuffer in the same frame!
	bool init(Device& device, bool renderToScreen);
	void deinit();

	void beginFrame();
	void endFrame();

	FenceValue renderFrame(const FrameData &frameData);

	void waitFence(FenceValue value);

	void onBackbufferResize();

	void setFramePipeline(FramePipeline *framePipeline);

	RenderSettings& getSettings() {
		return settings;
	}

	const RenderSettings& getSettings() const {
		return settings;
	}

	UINT64 getNumRenderedFrames() const {
		return numRenderedFrames;
	}

	UINT getBackbufferIndex() const {
		return backbufferIndex;
	}

	SizeType getNumPasses() const {
		if (framePipeline == nullptr) {
			return 0;
		}

		return framePipeline->getNumPasses();
	}

private:
	void renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle);

	CommandList populateCommandList(const FrameData &frameData);

private:
	FramePipeline *framePipeline;

	Device *device;

	UINT64 numRenderedFrames = 0;

	RenderSettings settings = {};

	// TODO: We don't use scissoring for now
	D3D12_RECT scissorRect = { 0,0, LONG_MAX, LONG_MAX };

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[FRAME_COUNT] = { 0, 0 };

	UINT backbufferIndex = 0; ///< Current backbuffer index

	bool allowTearing = false;

	/// Is this renderer outputs to the screen
	bool renderToScreen = false;
};

}