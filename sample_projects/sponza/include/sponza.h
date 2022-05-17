#pragma once

#include "async/job_system.h"
#include "d3d12/depth_buffer.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/vertex_index_buffer.h"
#include "framework/app.h"
#include "framework/camera.h"
#include "math/dar_math.h"
#include "utils/defines.h"

#include "fps_camera_controller.h"
#include "fps_edit_camera_controller.h"
#include "scene.h"

#include <functional>

struct Sponza : Dar::App {
	Sponza(UINT width, UINT height, const String &windowTitle);

private:
	bool loadAssets();

	// Inherited via D3D12App
	int initImpl() override;
	void deinit() override;
	void update() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseScroll(double xOffset, double yOffset) override;
	void onMouseMove(double xPos, double yPos) override;
	void onWindowClose() override;
	Dar::FrameData &getFrameData() override;

private:
	//CommandList populateCommandList();
	//void populateDeferredPassCommands(CommandList& cmdList);
	//void populateLightPassCommands(CommandList& cmdList);
	//void populateForwardPassCommands(CommandList& cmdList);
	//void populatePostPassCommands(CommandList &cmdList);
	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

	bool loadPipelines();
	bool prepareVertexIndexBuffers(Dar::UploadHandle);

private:
	using Super = Dar::App;

	enum class GBuffer : SizeType {
		Albedo = 0,
		Normals,
		MetallnessRoughnessOcclusion,
		Position,

		Count
	};

	const DXGI_FORMAT gBufferFormats[static_cast<SizeType>(GBuffer::Count)] = {
		DXGI_FORMAT_R8G8B8A8_UNORM, // Diffuse
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Normals
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Metalness+Roughness+Occlusion
		DXGI_FORMAT_R32G32B32A32_FLOAT // Position
	};

	enum class ProjectionType {
		Perspective,
		Orthographic
	} projectionType = ProjectionType::Perspective;

	Dar::VertexBuffer vertexBuffer;
	Dar::IndexBuffer indexBuffer;

	StaticArray<Dar::RenderTarget, static_cast<SizeType>(GBuffer::Count)> gBufferRTs;
	Dar::RenderTarget lightPassRT;

	Dar::DepthBuffer depthBuffer;

	Dar::FrameData frameData[Dar::FRAME_COUNT];

	// Const buffers
	Dar::ResourceHandle sceneDataHandle[Dar::FRAME_COUNT];

	float orthoDim = 10.f;

	// Scene
	Scene scene;

	// Root signature featyre level
	D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignatureFeatureData = {}; ///< Cache to the feature level for the root signature. Used when creating pipelines.

	FPSCameraController *camControl = nullptr;
	FPSCameraController fpsModeControl = { nullptr, 200.f };
	FPSEditModeCameraController editModeControl = { nullptr, 200.f };

	struct SponzaPassesArgs {
		Scene &scene;
		Dar::Renderer &renderer;
		Dar::DepthBuffer &dp;
		const DXGI_FORMAT *gBufferFormats;
		StaticArray<Dar::RenderTarget, static_cast<int>(GBuffer::Count)> &gBufferRTs;
		Dar::RenderTarget &lightPassRT;
	} args = { scene, renderer, depthBuffer, gBufferFormats, gBufferRTs, lightPassRT };

	// Debugging
	const char *gBufferLabels[7] = {"Render", "Diffuse", "Normals", "Metalness", "Roughness", "Occlusion", "Position"};
	bool editMode = true;
};
