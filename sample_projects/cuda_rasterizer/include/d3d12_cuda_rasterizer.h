#pragma once

#include "cuda.h"
#include "cuda_buffer.h"
#include "cuda_cpu_common.h"
#include "cuda_drawable.h"

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_resource_handle.h"

struct CUDAManager;

struct CudaRasterizer : D3D12App {
	CudaRasterizer(UINT width, UINT height, const String &windowTitle);
	
	int init() override;
	void deinit() override;

	virtual int loadScene(const String &name);

	/// Optional. \see D3D12App::drawUI()
	void drawUI() override;

	// API calls
	void setUseDepthBuffer(bool useDepthBuffer);
	void setVertexBuffer(const Vertex *buffer, SizeType verticesCount);
	void setIndexBuffer(const unsigned int *buffer, SizeType indicesCount);
	void setUAVBuffer(const void *buffer, SizeType size, int index);
	void setShaderProgram(const String &name);
	bool drawIndexed(const unsigned int numPrimitives);
	void setClearColor(Vec4 color);
	void setCulling(CudaRasterizerCullType cullType);
	void clearRenderTarget();
	void clearDepthBuffer();

private:
	int loadAssets();

	// Inherited via D3D12App
	void update() override;
	void render() override;
	virtual void onResize(int width, int height) override;
	virtual void onKeyboardInput(int key, int action) override;
	virtual void onMouseScroll(double xOffset, double yOffset) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();

	void timeIt();

	void deinitCUDAData();
	void initCUDAData();

private:
	using Super = D3D12App;

	static constexpr unsigned int numComps = 4;

	ComPtr<ID3D12RootSignature> rootSignature;
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
	float *cudaRenderTargetHost;
	ResourceHandle dx12RenderTargetHandle;
	CUDADefaultBuffer cudaRenderTargetDevice;
	CUDADefaultBuffer vertexBuffer;
	CUDADefaultBuffer indexBuffer;
	CUDADefaultBuffer depthBuffer;
	CUDADefaultBuffer uavBuffers[MAX_RESOURCES_COUNT];

	UINT64 fenceValues[frameCount];
	UINT64 previousFrameIndex;

	// Cache the cuda device we are using for rasterization
	const CUDADevice &cudaDevice;

	// TODO: something better than this
	Set<String> cachedShaders;

	Drawable *scene;

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
