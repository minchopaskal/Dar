#pragma once

#include "d3d12_app.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_scene.h"

struct Sponza : D3D12App {
	Sponza(UINT width, UINT height, const String &windowTitle);

	bool loadAssets();

	// Inherited via D3D12App
	int init() override;
	void deinit() override;
	void update() override;
	void render() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

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
	Mat4 mvp;
	ResourceHandle mvpBufferHandle[frameCount];

	// Texture data
	Vector<ResourceHandle> textureHandles;
	ComPtr<ID3D12DescriptorHeap> srvHeap;

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;
	float orthoDim = 10.f;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount];

	// Scene
	Scene scene;

	float FOV;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
