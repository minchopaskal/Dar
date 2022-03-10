#pragma once

#include "d3d12_app.h"
#include "d3d12_camera.h"
#include "d3d12_defines.h"
#include "d3d12_math.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_scene.h"

#include "fps_camera_controller.h"
#include "fps_edit_camera_controller.h"

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
	void populateDeferredPassCommands(CommandList& cmdList);
	void populateLightPassCommands(CommandList& cmdList);
	void populateForwardPassCommands(CommandList& cmdList);
	void populatePostPassCommands(CommandList &cmdList);
	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

	bool loadPipelines();
	bool loadVertexIndexBuffers();

	void timeIt();

private:
	using Super = D3D12App;

	enum class ProjectionType {
		Perspective,
		Orthographic
	} projectionType = ProjectionType::Perspective;

	enum class GBuffer {
		Diffuse,
		Specular,
		Normals,
		Position,

		Count
	};

	DXGI_FORMAT gBufferFormats[static_cast<SizeType>(GBuffer::Count)] = {
		DXGI_FORMAT_R8G8B8A8_UNORM, // Diffuse
		DXGI_FORMAT_R8G8B8A8_UNORM, // Specular
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Normals
		DXGI_FORMAT_R32G32B32A32_FLOAT // Position
	};

	// Descriptors
	PipelineState deferredPassPipelineState;
	DescriptorHeap deferredRTVHeap;
	DescriptorHeap deferredPassSRVHeap[frameCount];
	StaticArray<ResourceHandle, static_cast<SizeType>(GBuffer::Count) * frameCount> gBufferRTVTextureHandles;
	ResourceHandle vertexBufferHandle;
	ResourceHandle indexBufferHandle;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
	D3D12_INDEX_BUFFER_VIEW indexBufferView;
	DescriptorHeap dsvHeap;
	ResourceHandle depthBufferHandle;

	PipelineState lightPassPipelineState;
	DescriptorHeap lightPassRTVHeap;
	DescriptorHeap lightPassSRVHeap[frameCount];

	// Scene data handle
	ResourceHandle sceneDataHandle[frameCount];

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect;
	float aspectRatio;
	float orthoDim = 10.f;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount];

	// Scene
	Scene scene;

	FPSCameraController *camControl;
	FPSCameraController fpsModeControl;
	FPSEditModeCameraController editModeControl;

	bool editMode;

	// timing
	double fps;
	double totalTime;
	double deltaTime;

	// Debugging
	int showGBuffer;
};
