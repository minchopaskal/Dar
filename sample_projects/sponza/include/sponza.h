#pragma once

#include "framework/app.h"
#include "framework/camera.h"
#include "framework/scene.h"
#include "utils/defines.h"
#include "d3d12/depth_buffer.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/vertex_index_buffer.h"
#include "math/dar_math.h"

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
	bool prepareVertexIndexBuffers(UploadHandle);

	void timeIt();

private:
	using Super = D3D12App;

	enum class ProjectionType {
		Perspective,
		Orthographic
	} projectionType = ProjectionType::Perspective;

	enum class GBuffer {
		Albedo,
		Normals,
		MetallnessRoughnessOcclusion,
		Position,

		Count
	};

	DXGI_FORMAT gBufferFormats[static_cast<SizeType>(GBuffer::Count)] = {
		DXGI_FORMAT_R8G8B8A8_UNORM, // Diffuse
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Normals
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Metalness+Roughness+Occlusion
		DXGI_FORMAT_R32G32B32A32_FLOAT // Position
	};

	// Descriptors
	PipelineState deferredPassPipelineState;
	DescriptorHeap deferredRTVHeap;
	DescriptorHeap deferredPassSRVHeap[frameCount];
	StaticArray<ResourceHandle, static_cast<SizeType>(GBuffer::Count) * frameCount> gBufferRTVTextureHandles;

	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;

	DepthBuffer depthBuffer;

	PipelineState lightPassPipelineState;
	DescriptorHeap lightPassRTVHeap;
	DescriptorHeap lightPassSRVHeap[frameCount];
	StaticArray<ResourceHandle, frameCount> lightPassRTVTextureHandles;

	PipelineState postPassPipelineState;
	DescriptorHeap postPassRTVHeap;
	DescriptorHeap postPassSRVHeap[frameCount];

	// Scene data handle
	ResourceHandle sceneDataHandle[frameCount];

	// viewport
	D3D12_VIEWPORT viewport;
	D3D12_RECT scissorRect = { 0, 0, LONG_MAX, LONG_MAX };
	float aspectRatio;
	float orthoDim = 10.f;

	// Keeping track of fence values for double/triple buffering
	UINT64 fenceValues[frameCount] = { 0, 0 };

	// Scene
	Scene scene = { device };

	FPSCameraController *camControl = nullptr;
	FPSCameraController fpsModeControl = { nullptr, 200.f };
	FPSEditModeCameraController editModeControl = { nullptr, 200.f };


	// timing
	double fps = 0.0;
	double totalTime = 0.0;
	double deltaTime = 0.0;

	// Debugging
	const char *gBufferLabels[7] = {"Render", "Diffuse", "Normals", "Metalness", "Roughness", "Occlusion", "Position"};
	int showGBuffer = 0;
	bool spotLightOn = false;
	bool editMode = true;
	bool withNormalMapping = true;
};
