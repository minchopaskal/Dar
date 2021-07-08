#pragma once

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_resource_handle.h"

#include "cuda.h"
#include "cuda_buffer.h"

struct CUDAManager;

struct CudaRasterizer : D3D12App {
	CudaRasterizer(UINT width, UINT height, const String &windowTitle);

	// Inherited via D3D12App
	/// Derived classes should call CudaRasterizer::init()
	virtual int init() override;

	/// Optional. \see D3D12App::loadAssets()
	virtual int loadAssets() override;

	/// Derived classes should call CudaRasterizer::update()
	virtual void update() override;

	/// Derived classes have to call CudaRasterizer::render() at the end of their render() implementation
	virtual void render() override;
	
	/// Optional. \see D3D12App::drawUI()
	virtual void drawUI() override;

private:
	virtual void deinit() override;
	virtual void onResize(int width, int height) override;
	virtual void onKeyboardInput(int key, int action) override;
	virtual void onMouseScroll(double xOffset, double yOffset) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();

	void timeIt();

private:
	using Super = D3D12App;

	ComPtr<ID3D12RootSignature> rootSignature;
	//ComPtr<ID3D12PipelineState> pipelineState;
	PipelineState pipelineState;

	// Descriptors
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	ComPtr<ID3D12DescriptorHeap> srvHeap;
	UINT rtvHeapHandleIncrementSize;

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;

	// Resources
	UINT8 *cudaRT[frameCount];
	ResourceHandle rtHandle[frameCount];
	CUDADefaultBuffer renderTarget[frameCount];
	CUDADefaultBuffer color;

	UINT64 fenceValues[frameCount];
	UINT64 previousFrameIndex;

	// Cache cuda manager
	CUDAManager *cudaManager;

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
