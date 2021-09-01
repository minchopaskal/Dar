#pragma once

#include "cuda.h"
#include "cuda_buffer.h"
#include "cuda_cpu_common.h"

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_resource_handle.h"

struct CUDAManager;

struct CudaRasterizer : D3D12App {
	using DrawUICallback = void(*)();
	using UpdateFrameCallback = void(*)(CudaRasterizer &rasterizer, void *state);

	CudaRasterizer(Vector<String> &shadersFilenames, const String &windowTitle, UINT width, UINT height);
	~CudaRasterizer() override;

	void setUpdateFramebufferCallback(const UpdateFrameCallback cb, void *state);
	void setImGuiCallback(const DrawUICallback cb);

	bool isInitialized() const;

	// API calls
	CUDAError setUseDepthBuffer(bool useDepthBuffer);
	CUDAError setVertexBuffer(const Vertex* buffer, SizeType verticesCount);
	CUDAError setIndexBuffer(const unsigned int* buffer, SizeType indicesCount);
	CUDAError setUavBuffer(const void *buffer, SizeType size, int index);
	CUDAError setShaderProgram(const String &name) const;
	CUDAError drawIndexed(const unsigned int numPrimitives) const;
	CUDAError setClearColor(const Vec4 &color) const;
	CUDAError setCulling(CudaRasterizerCullType cullType) const;
	CUDAError clearRenderTarget();
	CUDAError clearDepthBuffer();

private:
	int loadAssets();
	int init() final;
	void deinit() final;

	// Inherited via D3D12App
	void update() override;
	void render() override;
	virtual void onResize(int width, int height) override;
	virtual void onKeyboardInput(int key, int action) override;
	virtual void onMouseScroll(double xOffset, double yOffset) override;
	void drawUI() override;

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

	// Cache the vertices count from the last setVertexBuffer call
	// in order to pass it to the processVertices kernel.
	// We could cache it on the gpu as a constant, but that's easier,
	// also, reduces constant memory reads.
	int cacheVerticesCount;

	UINT64 fenceValues[frameCount];
	UINT64 previousFrameIndex;

	// Cache the cuda device we are using for rasterization
	const CUDADevice *cudaDevice;

	// Callbacks
	UpdateFrameCallback updateFrameCb = nullptr;
	DrawUICallback drawUICb = nullptr;
	void *frameState = nullptr;

	// timing
	double fps;
	double totalTime;
	double deltaTime;

	bool inited;
};
