#pragma once

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_resource_handle.h"

#include "cuda.h"
#include "cuda_buffer.h"

struct CUDAManager;

#include "d3d12_math.h"

struct CudaRasterizer;
struct Drawable {
	virtual void draw(CudaRasterizer &renderer) = 0;
};

struct Vertex {
	Vec3 position;
	Vec3 normal;
	Vec2 uv;
};

struct Triangle {
	Vertex vertices[3];
};

struct TriangleDOD {
	Vec3 positions[3];
	/*Vec3 normals[3];
	Vec2 uvs[3];*/
};

struct Mesh : Drawable {
	void draw(CudaRasterizer &renderer) override;

	Vector<Triangle> geometry;
	Mat4 transform;
};

struct CudaRasterizer : D3D12App {
	CudaRasterizer(UINT width, UINT height, const String &windowTitle);
	
	int init() override;
	void deinit() override;

	int loadScene(const String &name);

	/// Optional. \see D3D12App::drawUI()
	void drawUI() override;

	// API calls
	void setUseDepthBuffer(bool useDepthBuffer);
	void setVertexBuffer(void *buffer);
	void setIndexBuffer(void *buffer);
	void setUAVBuffer(void *buffer, int index);
	void setVertexShader(const String &name);
	void setPixelShader(const String &name);

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

	Drawable *scene;

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
