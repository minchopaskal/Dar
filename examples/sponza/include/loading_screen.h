#pragma once

#include "frame_data.h"
#include "frame_pipeline.h"
#include "utils/defines.h"

#include "loading_screen_common.hlsli"

class LoadingScreen {
public:
	/// Initialize the loading screen state
	/// Subsequent calls do nothing.
	void init(Dar::Device &device);
	
	void deinit();

	void beginFrame();
	void endFrame();

	void render();

private:
	void uploadConstData();

private:
	Dar::Renderer renderer;
	Dar::FramePipeline pipeline;
	Dar::FrameData frameData[Dar::FRAME_COUNT];
	Dar::ResourceHandle constDataHandles[Dar::FRAME_COUNT];

	double timePassed = 0.;

	bool initialized = false;
};