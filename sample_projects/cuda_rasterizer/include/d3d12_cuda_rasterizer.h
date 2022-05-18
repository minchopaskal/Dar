#pragma once

#include "cuda.h"
#include "cuda_buffer.h"
#include "cuda_cpu_common.h"

#include "framework/app.h"
#include "utils/defines.h"
#include "math/dar_math.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/resource_handle.h"

struct CUDAManager;

struct CudaRasterizer : Dar::App {
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
	void deinit() final;
	
	// Inherited via D3D12App
	int initImpl() final;
	void update() override;
	virtual void onResize(const unsigned int w, const unsigned int h) override;
	virtual void onKeyboardInput(int key, int action) override;
	virtual void onMouseScroll(double xOffset, double yOffset) override;
	void drawUI() override;
	Dar::FrameData& getFrameData() override;

private:
	void deinitCUDAData();
	void initCUDAData();

private:
	using Super = Dar::App;

	static constexpr unsigned int numComps = 4;

	struct RenderPassArgs {
		Dar::TextureResource &texture;
		Dar::Renderer &renderer;
	} renderPassArgs = { dx12RT, renderer };

	float aspectRatio;

	// Resources
	float *cudaRenderTargetHost;
	Dar::TextureResource dx12RT;

	Dar::FrameData frameData[Dar::FRAME_COUNT];

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

	// Cache the cuda device we are using for rasterization
	const CUDADevice *cudaDevice;

	// Callbacks
	UpdateFrameCallback updateFrameCb = nullptr;
	DrawUICallback drawUICb = nullptr;
	void *frameState = nullptr;

	bool inited;
};
