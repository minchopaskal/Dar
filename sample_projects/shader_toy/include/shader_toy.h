#pragma once

#include "framework/app.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "graphics/d3d12/resource_handle.h"

struct ShaderToy : Dar::App {
	ShaderToy(UINT width, UINT height, const String &windowTitle);

private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	Dar::FrameData& getFrameData() override;

	int loadAssets();

private:
	using Super = Dar::App;

	Dar::FrameData frameData[Dar::FRAME_COUNT];
};
