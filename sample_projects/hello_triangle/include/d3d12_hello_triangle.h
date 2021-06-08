#pragma once

#include "d3d12_app.h"
#include "defines.h"

struct D3D12HelloTriangle : D3D12App {
	D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle);

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
	ComPtr<ID3D12GraphicsCommandList2> populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer(int width, int height);

	void timeIt();

private:
	ComPtr<ID3D12RootSignature> rootSignature;
	ComPtr<ID3D12PipelineState> pipelineState;

	// Descriptors
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	UINT rtvHeapHandleIncrementSize;
	ComPtr<ID3D12DescriptorHeap> dsvHeap;

	// Vertex buffer
	ComPtr<ID3D12Resource> vertexBuffer;
	ComPtr<ID3D12Resource> indexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	ComPtr<ID3D12Resource> depthBuffer;

	// MVP matrices
	Mat4 modelMat;
	Mat4 viewMat;
	Mat4 projectionMat;

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
