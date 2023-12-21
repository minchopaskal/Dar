#pragma once

#include "framework/app.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "graphics/d3d12/resource_handle.h"

struct HelloTriangle : Dar::App {
	HelloTriangle(UINT width, UINT height, const String &windowTitle);

private:
	// Inherited via D3D12App
	bool initImpl() override;
	void deinit() override;
	void update() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	Dar::FrameData& getFrameData();

	void beginFrame() override;
	void endFrame() override;

	int loadAssets();
	bool resizeDepthBuffer(int width, int height);

private:
	using Super = Dar::App;

	Dar::Renderer renderer;
	Dar::FramePipeline framePipeline;

	Dar::FrameData frameData[Dar::FRAME_COUNT];

	Dar::VertexBuffer vertexBuffer;
	Dar::IndexBuffer indexBuffer;
	Dar::DepthBuffer depthBuffer;

	Dar::DataBufferResource mvpResource[Dar::FRAME_COUNT];

	// MVP matrix
	Mat4 MVP;

	float aspectRatio;

	float FOV;
};
