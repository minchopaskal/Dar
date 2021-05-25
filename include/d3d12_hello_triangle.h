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
	void gpuSync();

private:
	static const UINT frameCount = 2;

	ComPtr<ID3D12CommandAllocator> commandAllocator;
	ComPtr<ID3D12GraphicsCommandList> commandList;
	ComPtr<ID3D12Device6> device;
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	ComPtr<ID3D12CommandQueue> commandQueue;
	ComPtr<IDXGISwapChain3> swapChain;
	ComPtr<ID3D12PipelineState> pipelineState;
	ComPtr<ID3D12Resource> renderTargets[frameCount];
	ComPtr<ID3D12RootSignature> rootSignature;
	
	// Vertex buffer
	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

	// Sync
	ComPtr<ID3D12Fence> fence;
	HANDLE fenceEvent;
	UINT64 fenceValue;

	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	real aspectRatio;

	UINT frameIndex;
	UINT rtvHeapHandleIncrementSize;
};
