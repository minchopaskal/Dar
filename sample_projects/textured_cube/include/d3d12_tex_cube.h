#pragma once

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"

struct D3D12TexturedCube : D3D12App {
	D3D12TexturedCube(UINT width, UINT height, const String &windowTitle);

	// Inherited via D3D12App
	virtual int init() override;
	virtual int loadAssets() override;
	virtual void deinit() override;
	virtual void update() override;
	virtual void render() override;
	virtual void onResize(int width, int height) override;
	virtual void onKeyboardInput(int key, int action) override;
	virtual void onMouseScroll(double xOffset, double yOffset) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer(int width, int height);

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
	Mat4 MVP;
	ResourceHandle mvpBufferHandle[frameCount];

	// Texture data
	static const int numTextures = 1;
	ResourceHandle texturesHandles[numTextures];
	ComPtr<ID3D12DescriptorHeap> srvHeap;

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;
	float orthoDim = 10.f;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount];

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
