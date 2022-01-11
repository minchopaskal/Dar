#pragma once

#include "d3d12_app.h"
#include "d3d12_camera.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_scene.h"

#include "fps_camera_controller.h"

struct Sponza : D3D12App {
	Sponza(UINT width, UINT height, const String &windowTitle);

	bool loadAssets();

private:
	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void render() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	void onMouseMove(double xPos, double yPos) override;

private:
	CommandList populateCommandList();
	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

	void setGLFWCursorHiddenState();

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

	// Camera data
	float FOV = 90.f;
	Vec3 camUp = { 0.f, 1.f, 0.f };
	Vec3 camForward = { 0.f, 0.f, 1.f };
	Vec3 camRight = { 1.f, 0.f, 0.f };
	Vec3 camPos = { 0.f, 0.f, 0.f };

	Camera cam;
	FPSCameraController camControl;

	bool cursorHidden;

	// timing
	double fps;
	double totalTime;
	double deltaTime;
};
