#pragma once

#include "framework/app.h"
#include "framework/camera.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "d3d12/pipeline_state.h"

#include "fps_camera_controller.h"

struct D3D12TexturedCube : D3D12App {
	D3D12TexturedCube(UINT width, UINT height, const String &windowTitle);

	int loadAssets();

private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void render() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	void onMouseMove(double xPos, double yPos) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

	void timeIt();

private:
	using Super = D3D12App;

	enum class ProjectionType {
		Perspective,
		Orthographic
	} projectionType = ProjectionType::Perspective;

	PipelineState pipelineState;

	// Descriptors
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	UINT rtvHeapHandleIncrementSize;
	ComPtr<ID3D12DescriptorHeap> dsvHeap;

	// Vertex buffer
	ResourceHandle vertexBufferHandle;
	ResourceHandle indexBufferHandle;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	ResourceHandle depthBufferHandle;

	// MVP matrix
	Mat4 mvp;
	ResourceHandle mvpBufferHandle[frameCount];

	// Texture data
	static constexpr int numTextures = 1;
	ResourceHandle texturesHandles[numTextures];
	ComPtr<ID3D12DescriptorHeap> srvHeap;

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;
	float orthoDim = 10.f;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount];

	// Camera
	Camera cam;
	FPSCameraController camControl;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
