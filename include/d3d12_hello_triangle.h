#pragma once

#include <string>

#include "d3d12_app.h"

#include "defines.h"

struct D3D12HelloTriangle : D3D12App {
	D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle);

	// Inherited via D3D12App
	virtual bool init() override;
	virtual void deinit() override;
	virtual void update() override;
	virtual void render() override;
	virtual void resize(int width, int height) override;

private:
	bool loadPipeline();
	bool loadAssets();

	void populateCommandList();

	UINT64 signal();
	void waitForFenceValue(UINT64 fenceVal);
	void flush();

	void timeIt();

	ComPtr<ID3D12CommandQueue> createCommandQueue(D3D12_COMMAND_LIST_TYPE type);
	ComPtr<ID3D12CommandAllocator> createCommandAllocator(D3D12_COMMAND_LIST_TYPE type);
	ComPtr<ID3D12GraphicsCommandList> createCommandList(ComPtr<ID3D12CommandAllocator> cmdAllocator, D3D12_COMMAND_LIST_TYPE type);
	bool updateRenderTargetViews();

private:
	static const UINT frameCount = 3;

	ComPtr<ID3D12Device6> device;

	ComPtr<ID3D12CommandQueue> commandQueueDirect;
	ComPtr<ID3D12CommandAllocator> commandAllocatorsDirect[frameCount];
	ComPtr<ID3D12GraphicsCommandList> commandListsDirect[frameCount];
	
	ComPtr<IDXGISwapChain4> swapChain;
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	ComPtr<ID3D12Resource> backBuffers[frameCount];
	
	ComPtr<ID3D12RootSignature> rootSignature;
	ComPtr<ID3D12PipelineState> pipelineState;

	// Vertex buffer
	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

	// Sync
	ComPtr<ID3D12Fence> fence;
	HANDLE fenceEvent;
	UINT64 fenceValue;
	UINT64 frameFenceValues[frameCount];

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	real aspectRatio;

	UINT frameIndex;
	UINT rtvHeapHandleIncrementSize;

	// timing
	double fps;
};
