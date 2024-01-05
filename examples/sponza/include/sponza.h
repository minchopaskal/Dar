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
#include "hud.h"
#include "loading_screen.h"
#include "scene.h"

#include "gpu_cpu_common.hlsli"

#include <functional>

struct Sponza : Dar::App {
	Sponza(UINT width, UINT height, const String &windowTitle);

private:
	class AppState {
	public:
		enum class State {
			Loading = 0,
			Game,
			Menu,

			Count
		};

		AppState(State initialState) : previous(initialState), current(initialState), newState(initialState) {}

		bool isState(State state) const {
			return current == state;
		}

		void setState(State state) {
			newState = state;
		}

		State getState() const {
			return current;
		}

		void beginFrame() {}

		void endFrame() {
			previous = current;
			if (newState != current) {
				current = newState;
			}
		}

		bool stateChanged() const {
			return previous != current;
		}
		
	private:
		State previous;
		State current;
		State newState;
	};

	bool loadAssets();

	// Inherited via D3D12App
	bool initImpl() override;
	void deinit() override;
	void update() override;
	void drawUI() override;
	void onResize(const unsigned int w, const unsigned int h) override;
	void onKeyboardInput(int key, int action) override;
	void onMouseButton(int button, int action, int mods) override;
	void onMouseMove(double xPos, double yPos) override;
	void onWindowClose() override;
	void beginFrame() override;
	void endFrame() override;

	bool updateRenderTargetViews();
	bool resizeDepthBuffer();

	void uploadShaderRenderData(Dar::UploadHandle);

	bool loadMainPipeline();
	bool prepareVertexIndexBuffers(Dar::UploadHandle);

	void updateLoadingScreen();
	void updateHUD();
	void updateMainLoop();

private:
	using Super = Dar::App;

	enum class GBuffer : SizeType {
		Albedo = 0,
		Normals,
		MetallnessRoughnessOcclusion,

		Count
	};

	const DXGI_FORMAT gBufferFormats[static_cast<SizeType>(GBuffer::Count)] = {
		DXGI_FORMAT_R8G8B8A8_UNORM, // Diffuse
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Normals
		DXGI_FORMAT_R32G32B32A32_FLOAT, // Metalness+Roughness+Occlusion
	};
	const DXGI_FORMAT lightPassRTFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;

	struct InitRes {
		Sponza *app;
		bool res;
	};

	struct HudJobParams {
		App *app;
		HUD *hud;
		AppState *state;
		WidgetHandle *quitButton;
		FenceValue *hudFence;
	} hudJobParams;

	Dar::Renderer renderer;
	Dar::FramePipeline mainPipeline;

	Dar::VertexBuffer vertexBuffer;
	Dar::IndexBuffer indexBuffer;

	StaticArray<Dar::RenderTarget, static_cast<SizeType>(GBuffer::Count)> gBufferRTs;
	Dar::RenderTarget lightPassRT;

	Dar::DepthBuffer depthBuffer;

	Dar::DepthBuffer shadowMapBuffer[MAX_SHADOW_MAPS_COUNT];

	Dar::FrameData frameData[Dar::FRAME_COUNT];

	AppState state{ AppState::State::Loading };
	InitRes initJobRes = {};
	Dar::JobSystem::Fence *initFence = nullptr;
	Dar::JobSystem::Fence *hudJobFence = nullptr;

	LoadingScreen loadingScreen;
	
	HUD hud;
	WidgetHandle quitButton;
	FenceValue hudRenderFence;

	Vec2 mousePos;

	double loadingDelta = 0.;
	double loadingTime = 0.;

	// Const buffers
	Dar::DataBufferResource sceneDataHandle[Dar::FRAME_COUNT];

	float orthoDim = 10.f;

	// Scene
	Scene scene;

	FPSCameraController *camControl = nullptr;
	FPSCameraController fpsModeControl = { nullptr, 200.f };
	FPSEditModeCameraController editModeControl = { nullptr, 200.f };

	// Debugging
	const char *gBufferLabels[9] = {"Render", "Diffuse", "Normals", "Metalness", "Roughness", "Occlusion", "Position", "Depth Map", "Shadow Map"};
	bool editMode;
	bool pause = false; ///< Pause all updates
};
