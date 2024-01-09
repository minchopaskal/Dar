#include "sponza.h"

#include <algorithm>
#include <cstdio>

#include "framework/app.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/resource_manager.h"
#include "utils/profile.h"
#include "utils/random.h"
#include "utils/utils.h"

#include "scene.h"
#include "scene_loader.h"

#include "reslib/resource_library.h"

#include "GLFW/glfw3.h" // keyboard input

Sponza::Sponza(const UINT w, const UINT h, const String &windowTitle) : Dar::App(w, h, windowTitle.c_str()) {
	editMode = false;
	camControl = editMode ? &editModeControl : &fpsModeControl;
}

void setGLFWCursorHiddenState(GLFWwindow *window, bool show) {
	glfwSetInputMode(window, GLFW_CURSOR, show ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

bool Sponza::initImpl() {
	// Load the shaders at first. They will be needed for the loading screen
	// Later we can load textures when needed.
	auto &resLibrary = Dar::getResourceLibrary();
	resLibrary.LoadShaderData();

	//glfwFocusWindow(getGLFWWindow());
	setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);

	auto initImplJob = [](void *param) {
		auto initRes = reinterpret_cast<InitRes *>(param);
		auto app = initRes->app;
		auto &res = initRes->res;
		LOG(Info, "Sponza::init");

		auto &resLibrary = Dar::getResourceLibrary();
		resLibrary.LoadTextureData();

		app->renderer.init(app->device, true /* renderToScreen */);

		app->fpsModeControl.window = app->editModeControl.window = app->getGLFWWindow();

		ComPtr<ID3D12Device> d12Device = app->device.getDevice();

		D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_6 };
		RETURN_ON_ERROR(
			d12Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)), ,
			"Device does not support shader model 6.6!"
		);

		if (shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_6) {
			LOG(Error, "Shader model 6.6 not supported!");
			res = false;
			return;
		}

		for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
			app->shadowMapBuffer[i].init(app->device.getDevice(), 2048, 2048, DXGI_FORMAT_D32_FLOAT, "ShadowMap[" + std::to_string(i) + "]");
		}

		if (!app->resizeDepthBuffer()) {
			res = false;
			return;
		}

		if (!app->updateRenderTargetViews()) {
			res = false;
			return;
		}

		app->hud.init(app->device);

		if (app->loadAssets()) {
			res = true;
			LOG(Info, "Sponza::init SUCCESS");
			return;
		}

		res = false;
	};

	initJobRes.app = this;
	Dar::JobSystem::JobDecl decl = {};
	decl.f = initImplJob;
	decl.param = &initJobRes;

	Dar::JobSystem::kickJobs(&decl, 1, &initFence);

	loadingScreen.init(device);

	return true;
}

void Sponza::deinit() {
	LOG(Info, "Sponza::deinit");
	
	flush();
	renderer.deinit();
	Super::deinit();

	LOG(Info, "Sponza::deinit SUCCESS");
}

void Sponza::update() {
	updateLoadingScreen();

	updateHUD();

	updateMainLoop();
}

void Sponza::updateMainLoop() {
	if (state.isState(AppState::State::Loading)) {
		return;
	}

	if (state.isState(AppState::State::Game)) {
		camControl->processKeyboardInput(this, getDeltaTime());

		if (!pause) {
			// Wanted to create an abstraction for scene node animations but would have taken too much time.
			LightData *pointLight = nullptr;
			for (int i = 0; i < scene.getNumLights(); ++i) {
				auto light = dynamic_cast<LightNode *>(scene.nodes[scene.lightIndices[i]]);
				if (light && light->lightData.type == LightType::Point) {
					pointLight = &light->lightData;
					//break;
				}
			}
			if (pointLight) {
				const float t = static_cast<float>(glm::sin(getTotalTime())) * 0.5f + 0.5f; // glm::smoothstep(0.01, 1.0, glm::abs(glm::sin(getTotalTime())));
				const Vec3 a = Vec3{ -1200.f, 175.f, 450.f };
				const Vec3 b = Vec3{ -1200.f, 175.f, -400.f };
				pointLight->position = a + t * (b - a);
			}
		}
		// any other updates...
	}

	auto uploadHandle = resManager->beginNewUpload();
	uploadShaderRenderData(uploadHandle);

	// TODO: If the app state is changed we need to disable using the same commands.
	const auto frameIndex = renderer.getBackbufferIndex();
	Dar::FrameData& fd = frameData[frameIndex];
	fd.setIndexBuffer(&indexBuffer);
	fd.setVertexBuffer(&vertexBuffer);
	fd.addConstResource(sceneDataHandle[frameIndex].getHandle(), static_cast<int>(DefaultConstantBufferView::SceneData));

	// Deferred pass:
	fd.startNewPass();
	scene.prepareFrameData(fd, uploadHandle);

	// Shadow map pass:
	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		fd.startNewPass();
		scene.prepareFrameDataForShadowMap(i, fd, uploadHandle);
	}

	// Lighting pass:
	fd.startNewPass();
	fd.addDataBufferResource(scene.lightsBuffer);
	fd.addTextureResource(depthBuffer.getTexture());
	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		fd.addTextureResource(shadowMapBuffer[i].getTexture());
	}
	for (int i = 0; i < static_cast<UINT>(GBuffer::Count); ++i) {
		fd.addTextureResource(gBufferRTs[i].getTextureResource(frameIndex));
	}
	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0));

	// Post-process pass
	fd.startNewPass();
	fd.addTextureResource(lightPassRT.getTextureResource(frameIndex));
	fd.addTextureResource(depthBuffer.getTexture());
	fd.addTextureResource(hud.getTexture());
	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0));

	if (renderer.getNumRenderedFrames() < Dar::FRAME_COUNT) {
		fd.setUseSameCommands(true);
	}

	Dar::JobSystem::waitFenceAndFree(hudJobFence);
	auto uploadCtx = resManager->uploadBuffersAsync();
	fd.addUploadContextToWait(uploadCtx);
	fd.addFenceToWait(hudRenderFence);
	
	renderer.renderFrame(fd);
}

void Sponza::drawUI() {
	if (state.isState(AppState::State::Loading)) {
		return;
	}

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);

	const Dar::Camera &cam = camControl->getCamera();

	ImGui::SetNextWindowPos({ 0, 0 });

	ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("FPS: %.2f", getFPS());
		ImGui::Text("Frame Time: %.2f", getFrameTime());
		ImGui::Text("Camera FOV: %.2f", cam.getFOV());
		ImGui::Text("Camera Speed: %.2f", camControl->getSpeed());
		Vec3 pos = cam.getPos();
		ImGui::Text("Camera Position: %.2f %.2f %.2f", pos.x, pos.y, pos.z);
		ImGui::Text("Camera Vectors:");
		Vec3 x = cam.getCameraX();
		Vec3 y = cam.getCameraY();
		Vec3 z = cam.getCameraZ();
		ImGui::Text("Right: %.2f %.2f %.2f", x.x, x.y, x.z);
		ImGui::Text("Up: %.2f %.2f %.2f", y.x, y.y, y.z);
		ImGui::Text("Forward: %.2f %.2f %.2f", z.x, z.y, z.z);
		ImGui::GetWindowHeight();
		ImVec2 winPos = ImGui::GetWindowPos();
		ImVec2 winSize = ImGui::GetWindowSize();
	ImGui::End();

	ImGui::SetNextWindowPos({ winPos.x, winPos.y + winSize.y });
	ImGui::Begin("General controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("[`] - Show Rendered image");
		ImGui::Text("[1-6] - Show G-Buffers");
		ImGui::Text("[7] - Show Depth Map");
		ImGui::Text("[8] - Show Shadow Map");
		ImGui::Text("[m] - Switch between FPS/edit modes");
		ImGui::Text("[f] - Toggle spotlight");
		ImGui::Text("[p] - Toggle fullscreen mode");
		ImGui::Text("[v] - Toggle V-Sync mode");
		winPos = ImGui::GetWindowPos();
		winSize = ImGui::GetWindowSize();
	ImGui::End();

	ImGui::SetNextWindowPos({ winPos.x, winPos.y + winSize.y });
	ImGui::Begin("FPS Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("[mouse move] - Turn around");
		ImGui::Text("[wasd] - Move forwards/left/backwards/right");
		ImGui::Text("[qe] - Move up/down");
		ImGui::Text("[rt] - Increase/Decrease camera speed");
		ImGui::Text("[shift] - Hold to move twice as fast.");
		winPos = ImGui::GetWindowPos();
		winSize = ImGui::GetWindowSize();
	ImGui::End();

	ImGui::SetNextWindowPos({ winPos.x, winPos.y + winSize.y });
	ImGui::Begin("FPS Edit Mode Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("[alt] - Hold for movement and rotation of camera");
		ImGui::Text("[mouse scroll] - Zoom/unzoom");
		winPos = ImGui::GetWindowPos();
		winSize = ImGui::GetWindowSize();
	ImGui::End();

	Dar::RenderSettings &rs = renderer.getSettings();
	
	if (editMode) {
		static float editModeWinWidth = 0.f;

		ImGui::SetNextWindowPos({ width - editModeWinWidth, 0 });
		const ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
		
		ImGui::Begin("Edit mode", nullptr, flags);
			ImGui::ListBox("G-Buffer", &rs.showGBuffer, gBufferLabels, sizeof(gBufferLabels) / sizeof(char *));
			ImGui::Checkbox("With normal mapping", &rs.enableNormalMapping);
			ImGui::Checkbox("V-Sync", &renderer.getSettings().vSyncEnabled);
			ImGui::Checkbox("FXAA", &rs.enableFXAA);
			editModeWinWidth = ImGui::GetWindowWidth();
		ImGui::End();
	}
}

void Sponza::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	width = std::max(1u, w);
	height = std::max(1u, h);

	scene.getRenderCamera()->updateAspectRatio(width, height);

	flush();

	device.resizeBackBuffers();

	const int gBufferCount = static_cast<UINT>(GBuffer::Count);
	for (int j = 0; j < gBufferCount; ++j) {
		gBufferRTs[j].resizeRenderTarget(width, height);
	}

	lightPassRT.resizeRenderTarget(width, height);

	resizeDepthBuffer();

	hud.resize();
}

void Sponza::onKeyboardInput(int /*key*/, int /*action*/) {
	if (state.isState(AppState::State::Loading)) {
		return;
	}

	if (queryPressed(GLFW_KEY_SPACE)) {
		pause = !pause;
	}

	if (queryPressed(GLFW_KEY_ESCAPE)) {
		if (state.isState(AppState::State::Menu)) {
			editModeControl.setMouseDisabled(false);
			fpsModeControl.setMouseDisabled(false);
			setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);
			state.setState(AppState::State::Game);
		}
		if (state.isState(AppState::State::Game)) {
			editModeControl.setMouseDisabled(true);
			fpsModeControl.setMouseDisabled(true);
			setGLFWCursorHiddenState(getGLFWWindow(), true);
			state.setState(AppState::State::Menu);
		}
	}

	if (queryPressed(GLFW_KEY_M)) {
		editMode = !editMode;
		renderer.getSettings().useImGui = editMode == true;
		setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);
		camControl = editMode ? &editModeControl : &fpsModeControl;
	}

	if (state.isState(AppState::State::Menu)) {
		return;
	}

	if (keyPressed[GLFW_KEY_P] && !keyRepeated[GLFW_KEY_P]) {
		toggleFullscreen();
	}

	Dar::RenderSettings &rs = renderer.getSettings();

	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		rs.spotLightON = !rs.spotLightON;
	}

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		rs.vSyncEnabled = !rs.vSyncEnabled;
	}

	if (queryPressed(GLFW_KEY_GRAVE_ACCENT)) {
		rs.showGBuffer = 0;
	}

	for (int i = GLFW_KEY_0; i <= GLFW_KEY_9; ++i) {
		if (queryPressed(i)) {
			rs.showGBuffer = i - GLFW_KEY_0;
		}
	}
}

void Sponza::onMouseButton(int button, int action, int /*mods*/) {
	if (state.isState(AppState::State::Menu)) {
		if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
			Vec2 pos{ mousePos.x / getWidth(), mousePos.y / getHeight() };
			if (hud.isOverButton(quitButton, pos)) {
				quit();
			}
		}
	}
}

void Sponza::onMouseMove(double xPos, double yPos) {
	mousePos = Vec2{ float(xPos), float(yPos) };

	if (state.isState(AppState::State::Game)) {
		camControl->onMouseMove(xPos, yPos, getDeltaTime());
	}
}

void Sponza::onWindowClose() {
	abort = 1;
}

void Sponza::beginFrame() {
	App::beginFrame();

	state.beginFrame();

	if (state.isState(AppState::State::Loading)) {
		loadingScreen.beginFrame();
	} else {
		hud.beginFrame();
		renderer.beginFrame();
		frameData[renderer.getBackbufferIndex()].beginFrame(renderer);
	}
}

void Sponza::endFrame() {
	if (state.isState(AppState::State::Loading)) {
		loadingScreen.endFrame();
	} else {
		frameData[renderer.getBackbufferIndex()].endFrame(renderer);
		renderer.endFrame();
		hud.endFrame();
	}

	state.endFrame();
}

bool Sponza::loadAssets() {
	LOG_FMT(Info, "Sponza::loadAssets");

	if (!loadMainPipeline()) {
		LOG_FMT(Error, "Failed to load Main pipeline!");
		return false;
	}

	LOG_FMT(Info, "Sponza::loadScene");
	// TODO: load the scene in binary format + run MikkTSpace on the tangents in a preprocess step.
	//       During runtime we should only read the bin scene and upload the data to the gpu.
	// MikkTSpace tangents give slightly better results than the tangents in the gltf file.
	//SceneLoaderError sceneLoadErr = loadScene("res\\scenes\\Sponza\\glTF\\Sponza.gltf", scene, sceneLoaderFlags_overrideGenTangents);
	// .. but the algorithm is too slow for run-time evaluation.
	SceneLoaderError sceneLoadErr = loadScene("res\\scenes\\sponza.json", scene, sceneLoaderFlags_none);
	LOG_FMT(Info, "Sponza::loadScene SUCCESS");

	if (sceneLoadErr != SceneLoaderError::Success) {
		LOG(Error, "Failed to load scene!");
		return false;
	}

	bool setCamRes = scene.setCameraForCameraController(fpsModeControl);
	if (!setCamRes) {
		LOG(Error, "Failed to set FPS camera controller!");
		return false;
	}
	setCamRes = scene.setCameraForCameraController(editModeControl);
	if (!setCamRes) {
		LOG(Error, "Failed to set Edit mode camera controller!");
		return false;
	}

	Dar::UploadHandle uploadHandle = resManager->beginNewUpload();

	if (!scene.uploadSceneData(uploadHandle)) {
		LOG(Error, "Failed to upload scene data!");
		return false;
	}

	if (!prepareVertexIndexBuffers(uploadHandle)) {
		LOG(Error, "Failed to prepare vertex and index buffers!");
		return false;
	}

	resManager->uploadBuffers();

	LOG_FMT(Info, "Sponza::loadAssets SUCCESS");
	return true;
}

bool Sponza::updateRenderTargetViews() {
	auto getGBufferName = [](GBuffer type) -> String {
		String res;
		switch (type) {
		case GBuffer::Albedo:
			res = "GBuffer::Albedo";
			break;
		case GBuffer::Normals:
			res = "GBuffer::Normals";
			break;
		case GBuffer::MetallnessRoughnessOcclusion:
			res = "GBuffer::MetallnessRoughnessOcclusion";
			break;
		default:
			res = "GBuffer::Unknown";
			break;
		}

		return res;
	};

	const int gBufferCount = static_cast<UINT>(GBuffer::Count);
	for (int i = 0; i < gBufferCount; ++i) {
		Dar::TextureInitData rtvTextureDesc = {};
		rtvTextureDesc.width = width;
		rtvTextureDesc.height = height;
		rtvTextureDesc.format = gBufferFormats[i];
		rtvTextureDesc.clearValue.color[0] = 0.f;
		rtvTextureDesc.clearValue.color[1] = 0.f;
		rtvTextureDesc.clearValue.color[2] = 0.f;
		rtvTextureDesc.clearValue.color[3] = 0.f;

		gBufferRTs[i].init(rtvTextureDesc, Dar::FRAME_COUNT, getGBufferName(static_cast<GBuffer>(i)));
	}

	String lightPassRTName = "LightPassRTV";
	Dar::TextureInitData rtvTextureDesc = {};
	rtvTextureDesc.width = width;
	rtvTextureDesc.height = height;
	rtvTextureDesc.format = lightPassRTFormat;

	lightPassRT.init(rtvTextureDesc, Dar::FRAME_COUNT, lightPassRTName);

	return true;
}

bool Sponza::resizeDepthBuffer() {
	return depthBuffer.init(device.getDevice(), getWidth(), getHeight(), DXGI_FORMAT_D32_FLOAT, "DepthBuffer");
}

void Sponza::uploadShaderRenderData(Dar::UploadHandle uploadHandle) {
	const Dar::RenderSettings& rs = renderer.getSettings();
	const Dar::Camera& cam = camControl->getCamera();

	ShaderRenderData sceneData = {};
	Mat4 viewMat = cam.getViewMatrix();
	Mat4 projectionMat = cam.getProjectionMatrix();
	sceneData.viewProjection = projectionMat * viewMat;
	sceneData.invView = glm::inverse(viewMat);
	sceneData.invProjection = glm::inverse(projectionMat);
	sceneData.cameraPosition = Vec4{ cam.getPos(), 1.f };
	sceneData.cameraDir = Vec4{ glm::normalize(cam.getCameraZ()), 1.f };
	sceneData.numLights = static_cast<int>(scene.getNumLights());
	sceneData.showGBuffer = rs.showGBuffer;
	sceneData.withNormalMapping = rs.enableNormalMapping;
	sceneData.spotLightON = rs.spotLightON;
	sceneData.fxaaON = rs.enableFXAA;
	sceneData.invWidth = 1.f / width;
	sceneData.invHeight = 1.f / height;
	sceneData.nearPlane = cam.getNearPlane();
	sceneData.farPlane = cam.getFarPlane();
	sceneData.width = width;
	sceneData.height = height;
	sceneData.time = static_cast<float>(getTotalTime());
	sceneData.delta = static_cast<float>(getDeltaTime());
	sceneData.frame = static_cast<UINT>(renderer.getNumRenderedFrames()); // see LoadingScreen::uploadConstData
	sceneData.darken = state.isState(AppState::State::Menu);

	/// Initialize the MVP constant buffer resource if needed
	const int frameIndex = renderer.getBackbufferIndex();
	sceneDataHandle[frameIndex].init(sizeof(ShaderRenderData), 1);

	sceneDataHandle[frameIndex].upload(uploadHandle, &sceneData);
}

bool Sponza::loadMainPipeline() {
	LOG_FMT(Info, "Sponza::loadMainPipeline");

	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	D3D12_STATIC_SAMPLER_DESC staticSamplers[]{
		CD3DX12_STATIC_SAMPLER_DESC{ 0, D3D12_FILTER_ANISOTROPIC },
		CD3DX12_STATIC_SAMPLER_DESC{
			1,
			D3D12_FILTER_MIN_MAG_MIP_POINT,
			D3D12_TEXTURE_ADDRESS_MODE_BORDER,
			D3D12_TEXTURE_ADDRESS_MODE_BORDER,
			D3D12_TEXTURE_ADDRESS_MODE_BORDER,
			0.f,
			16,
			D3D12_COMPARISON_FUNC_GREATER,
			D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK
		},
	};
	
	Dar::RenderPassDesc deferredPassDesc = {};
	Dar::PipelineStateDesc deferredPSDesc = {};
	deferredPSDesc.shaderName = "deferred";
	deferredPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	deferredPSDesc.inputLayouts = inputLayouts;
	deferredPSDesc.staticSamplerDescs = staticSamplers;
	deferredPSDesc.numStaticSamplers = _countof(staticSamplers);
	deferredPSDesc.numInputLayouts = _countof(inputLayouts);
	deferredPSDesc.depthStencilBufferFormat = depthBuffer.getFormatAsDepthBuffer();
	deferredPSDesc.numConstantBufferViews = static_cast<UINT>(DefaultConstantBufferView::Count);
	deferredPSDesc.numRenderTargets = static_cast<UINT>(GBuffer::Count);
	for (UINT i = 0; i < deferredPSDesc.numRenderTargets; ++i) {
		deferredPSDesc.renderTargetFormats[i] = gBufferFormats[i];
	}
	deferredPassDesc.setPipelineStateDesc(deferredPSDesc);
	for (int i = 0; i < static_cast<int>(GBuffer::Count); ++i) {
		deferredPassDesc.attach(Dar::RenderPassAttachment::renderTarget(&gBufferRTs[i]));
	}
	deferredPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&depthBuffer, true));
	mainPipeline.addRenderPass(deferredPassDesc);

	for (int i = 0; i < MAX_SHADOW_MAPS_COUNT; ++i) {
		Dar::RenderPassDesc shadowMapPassDesc = {};
		Dar::PipelineStateDesc shadowMapPSDesc = {};
		shadowMapPSDesc.shaderName = "shadow_map";
		shadowMapPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
		shadowMapPSDesc.inputLayouts = inputLayouts;
		shadowMapPSDesc.staticSamplerDescs = staticSamplers;
		shadowMapPSDesc.numStaticSamplers = _countof(staticSamplers);
		shadowMapPSDesc.numInputLayouts = _countof(inputLayouts);
		shadowMapPSDesc.depthStencilBufferFormat = shadowMapBuffer[i].getFormatAsDepthBuffer();
		shadowMapPSDesc.numConstantBufferViews = static_cast<UINT>(ShadowMapConstantBufferView::Count);
		shadowMapPSDesc.numRenderTargets = 0;
		shadowMapPassDesc.setPipelineStateDesc(shadowMapPSDesc);
		shadowMapPassDesc.setViewport(shadowMapBuffer[i].getTexture().getWidth(), shadowMapBuffer[i].getTexture().getHeight());
		shadowMapPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&shadowMapBuffer[i], true));
		mainPipeline.addRenderPass(shadowMapPassDesc);
	}

	Dar::RenderPassDesc lightingPassDesc = {};
	Dar::PipelineStateDesc lightingPSDesc = {};
	lightingPSDesc.shaderName = "lighting";
	lightingPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	lightingPSDesc.staticSamplerDescs = staticSamplers;
	lightingPSDesc.numStaticSamplers = _countof(staticSamplers);
	lightingPSDesc.numConstantBufferViews = static_cast<unsigned int>(DefaultConstantBufferView::Count);
	lightingPSDesc.cullMode = D3D12_CULL_MODE_NONE;
	lightingPSDesc.renderTargetFormats[0] = lightPassRT.getFormat();
	lightingPassDesc.setPipelineStateDesc(lightingPSDesc);
	lightingPassDesc.attach(Dar::RenderPassAttachment::renderTarget(&lightPassRT));
	mainPipeline.addRenderPass(lightingPassDesc);

	Dar::RenderPassDesc postPassDesc = {};
	Dar::PipelineStateDesc postPSDesc = {};
	postPSDesc.shaderName = "post";
	postPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	postPSDesc.staticSamplerDescs = staticSamplers;
	postPSDesc.numStaticSamplers = _countof(staticSamplers);
	postPSDesc.numConstantBufferViews = static_cast<unsigned int>(DefaultConstantBufferView::Count);
	postPSDesc.cullMode = D3D12_CULL_MODE_NONE;
	postPassDesc.setPipelineStateDesc(postPSDesc);
	postPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	mainPipeline.addRenderPass(postPassDesc);

	if (!mainPipeline.compilePipeline(device)) {
		return false;
	}
	renderer.setFramePipeline(&mainPipeline);

	LOG_FMT(Info, "Sponza::loadMainPipeline SUCCESS");
	return true;
}

bool Sponza::prepareVertexIndexBuffers(Dar::UploadHandle uploadHandle) {
	LOG_FMT(Info, "Sponza::prepareVertexIndexBuffers");

	Dar::VertexIndexBufferDesc vertexDesc = {};
	vertexDesc.data = scene.getVertexBuffer();
	vertexDesc.size = scene.getVertexBufferSize();
	vertexDesc.name = "VertexBuffer";
	vertexDesc.vertexBufferStride = sizeof(Vertex);
	if (!vertexBuffer.init(vertexDesc, uploadHandle)) {
		return false;
	}

	Dar::VertexIndexBufferDesc indexDesc = {};
	indexDesc.data = scene.getIndexBuffer();
	indexDesc.size = scene.getIndexBufferSize();
	indexDesc.name = "IndexBuffer";
	indexDesc.indexBufferFormat = DXGI_FORMAT_R32_UINT;
	if (!indexBuffer.init(indexDesc, uploadHandle)) {
		return false;
	}

	LOG_FMT(Info, "Sponza::prepareVertexIndexBuffers SUCCESS");
	return true;
}

void Sponza::updateLoadingScreen() {
	if (state.isState(AppState::State::Loading)) {
		if (!Dar::JobSystem::probeFence(initFence)) {
			loadingDelta += getDeltaTime();
			loadingTime += getDeltaTime();
			if (loadingDelta > 0.1) {
				loadingDelta = 0.;
				LOG_FMT(Info, "Loading... %f", loadingTime);
			}

			loadingScreen.render();

			return;
		}

		Dar::JobSystem::waitFenceAndFree(initFence);

		if (!initJobRes.res) {
			quit();
		}

		state.setState(AppState::State::Game);

		return;
	}
}

void Sponza::updateHUD() {
	if (state.isState(AppState::State::Loading)) {
		return;
	}

	hudRenderFence = 0;
	hudJobParams = { this, &hud, &state, &quitButton, &hudRenderFence };

	Dar::JobSystem::JobDecl hudJob = {};
	hudJob.f = [](void *param) {
		auto params = reinterpret_cast<HudJobParams *>(param);
		auto app = params->app;
		auto &hud = *params->hud;
		auto &state = *params->state;
		auto &quitButton = *params->quitButton;
		auto &fenceValue = *params->hudFence;

		switch (state.getState()) {
		case AppState::State::Loading:
			return;
		case AppState::State::Game:
		{
			Vec2 hudTopLeft = { .05f, .05f };
			Vec2 hudDims = { 0.1f, 0.05 * app->getWidth() / float(app->getHeight()) };
			auto rectWidget = RectWidgetDesc{
				.texture = "in_game_hud.png",
				.topLeft = hudTopLeft,
				.size = hudDims,
				.depth = 0
			};
			hud.addRectWidget(rectWidget);
			break;
		}
		case AppState::State::Menu:
		{
			Vec2 hudDims = { 0.1f, 0.05 * app->getWidth() / float(app->getHeight()) };
			Vec2 hudTopLeft = { .5f - hudDims.x / 2., .5f - hudDims.y / 2. };
			auto rectWidget = RectWidgetDesc{
				.texture = "quit_button.png",
				.topLeft = hudTopLeft,
				.size = hudDims,
				.depth = 0
			};
			quitButton = hud.addButton(rectWidget);
			break;
		}
		default:
			dassert(false);
			break;
		}

		fenceValue = hud.render();
	};
	hudJob.param = &hudJobParams;
	Dar::JobSystem::kickJobs(&hudJob, 1, &hudJobFence);
}
