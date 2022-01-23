#pragma once

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_resource_handle.h"

struct D3D12HelloTriangle : D3D12App {
	D3D12HelloTriangle(UINT width, UINT height, const String &windowTitle);

	int loadAssets();

private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void render() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer(int width, int height);

	void timeIt();

private:
	using Super = D3D12App;

	ComPtr<ID3D12RootSignature> rootSignature;
	ComPtr<ID3D12PipelineState> pipelineState;

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

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount];

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
