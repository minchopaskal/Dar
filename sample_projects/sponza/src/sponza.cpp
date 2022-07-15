#include "sponza.h"

#include <algorithm>
#include <cstdio>

#include "framework/app.h"
#include "asset_manager/asset_manager.h"
#include "d3d12/pipeline_state.h"
#include "d3d12/resource_manager.h"
#include "utils/profile.h"
#include "utils/random.h"
#include "utils/utils.h"

#include "scene.h"
#include "scene_loader.h"

#include "gpu_cpu_common.hlsli"

#include "GLFW/glfw3.h" // keyboard input

Sponza::Sponza(const UINT w, const UINT h, const String &windowTitle) : Dar::App(w, h, windowTitle.c_str()) {
	camControl = editMode ? &editModeControl : &fpsModeControl;
}

void setGLFWCursorHiddenState(GLFWwindow *window, bool show) {
	glfwSetInputMode(window, GLFW_CURSOR, show ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

int Sponza::initImpl() {
	setUseImGui();

	fpsModeControl.window = editModeControl.window = getGLFWWindow();

	setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);

	ComPtr<ID3D12Device> device = renderer.getDevice();

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_6 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.6!"
	);

	RETURN_ERROR_IF(shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_6, false, "Shader model 6.6 not supported!");

	if (!resizeDepthBuffer()) {
		return false;
	}

	if (!updateRenderTargetViews()) {
		return false;
	}

	return loadAssets();
}

void Sponza::deinit() {
	flush();
	Super::deinit();
}

void Sponza::update() {
	/* Simulate some work to test the fiber job system. */
	{
		auto busyWork = [](void *param) {
			DAR_OPTICK_EVENT("Busy work");
			const SizeType n = reinterpret_cast<SizeType>(param);
			for (int i = 0; i < n; ++i) {
				const int a = 5000 * 50000;
				const double b = sqrt(pow(a, 2.2));
			}
		};

		Dar::JobSystem::JobDecl jobs[100];
		for (SizeType i = 0; i < 100; ++i) {
			jobs[i].f = busyWork;
			jobs[i].param = reinterpret_cast<void *>(i * 1000);
		}

		//Dar::JobSystem::kickJobsAndWait(jobs, 100);
	}

	camControl->processKeyboardInput(this, getDeltaTime());

	const Dar::Camera &cam = camControl->getCamera();

	// Update VP matrices
	Mat4 viewMat = cam.getViewMatrix();
	Mat4 projectionMat = cam.getProjectionMatrix();

	const Dar::RenderSettings &rs = renderer.getSettings();

	ShaderRenderData sceneData = {};
	sceneData.viewProjection = projectionMat * viewMat;
	sceneData.cameraPosition = Vec4{ cam.getPos(), 1.f };
	sceneData.cameraDir = Vec4{ dmath::normalized(cam.getCameraZ()), 1.f };
	sceneData.invWidth = 1.f / width;
	sceneData.invHeight = 1.f / height;
	sceneData.numLights = static_cast<int>(scene.getNumLights());
	sceneData.showGBuffer = rs.showGBuffer;
	sceneData.width = width;
	sceneData.height = height;
	sceneData.withNormalMapping = rs.enableNormalMapping;
	sceneData.spotLightON = rs.spotLightON;
	sceneData.fxaaON = rs.enableFXAA;

	/// Initialize the MVP constant buffer resource if needed
	const int frameIndex = renderer.getBackbufferIndex();
	if (sceneDataHandle[frameIndex] == INVALID_RESOURCE_HANDLE) {
		wchar_t frameMVPName[32] = L"";
		swprintf(frameMVPName, 32, L"SceneData[%d]", frameIndex);

		Dar::ResourceInitData resData(Dar::ResourceType::DataBuffer);
		resData.size = sizeof(ShaderRenderData);
		resData.name = frameMVPName;
		sceneDataHandle[frameIndex] = resManager->createBuffer(resData);
	}

	Dar::UploadHandle uploadHandle = resManager->beginNewUpload();
	resManager->uploadBufferData(uploadHandle, sceneDataHandle[frameIndex], reinterpret_cast<void*>(&sceneData), sizeof(ShaderRenderData));
	resManager->uploadBuffers();

	Dar::FrameData &fd = frameData[frameIndex];
	fd.setIndexBuffer(&indexBuffer);
	fd.setVertexBuffer(&vertexBuffer);
	fd.addConstResource(sceneDataHandle[frameIndex], 0);

	// Deferred pass:
	scene.prepareFrameData(fd);

	// Lighting pass:
	fd.addDataBufferResource(scene.lightsBuffer, 1);
	fd.addTextureResource(depthBuffer.getTexture(), 1);
	for (int i = 0; i < static_cast<UINT>(GBuffer::Count); ++i) {
		fd.addTextureResource(gBufferRTs[i].getTextureResource(frameIndex), 1);
	}

	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0), 1);

	// Post-process pass
	fd.addTextureResource(lightPassRT.getTextureResource(frameIndex), 2);
	fd.addTextureResource(depthBuffer.getTexture(), 2);

	fd.addRenderCommand(Dar::RenderCommandDrawInstanced(3, 1, 0, 0), 2);

	if (renderer.getNumRenderedFrames() < Dar::FRAME_COUNT) {
		fd.setUseSameCommands(true);
	}
}

void Sponza::drawUI() {
	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);

	const Dar::Camera &cam = camControl->getCamera();

	ImGui::SetNextWindowPos({ 0, 0 });

	ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("FPS: %.2f", getFPS());
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
		ImGui::Text("[1-4] - Show G-Buffers");
		ImGui::Text("[m] - Switch between FPS/edit modes");
		ImGui::Text("[o] - Switch between perspective/orthographic projection");
		ImGui::Text("[f] - Toggle fullscreen mode");
		ImGui::Text("[v] - Toggle V-Sync mode");
		winPos = ImGui::GetWindowPos();
		winSize = ImGui::GetWindowSize();
	ImGui::End();

	ImGui::SetNextWindowPos({ winPos.x, winPos.y + winSize.y });
	ImGui::Begin("FPS Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("[mouse move] - Turn around");
		ImGui::Text("[mouse scroll] - Zoom/unzoom");
		ImGui::Text("[wasd] - Move forwards/left/backwards/right");
		ImGui::Text("[qe] - Move up/down");
		ImGui::Text("[rt] - Increase/Decrease camera speed");
		ImGui::Text("[k] - Make/Stop camera keeping on the plane of walking");
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
		ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
		
		ImGui::Begin("Edit mode", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
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

	this->width = std::max(1u, w);
	this->height = std::max(1u, h);

	scene.getRenderCamera()->updateAspectRatio(width, height);

	flush();

	renderer.resizeBackBuffers();

	const int gBufferCount = static_cast<UINT>(GBuffer::Count);
	for (int j = 0; j < gBufferCount; ++j) {
		gBufferRTs[j].resizeRenderTarget(width, height);
	}

	lightPassRT.resizeRenderTarget(width, height);

	resizeDepthBuffer();
}

void Sponza::onKeyboardInput(int key, int action) {
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

	if (keyPressed[GLFW_KEY_O] && !keyRepeated[GLFW_KEY_O]) {
		projectionType = static_cast<ProjectionType>((static_cast<int>(projectionType) + 1) % 2);
	}

	if (queryPressed(GLFW_KEY_M)) {
		editMode = !editMode;
		setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);
		camControl = editMode ? &editModeControl : &fpsModeControl;
	}

	if (queryPressed(GLFW_KEY_GRAVE_ACCENT)) {
		rs.showGBuffer = 0;
	}

	if (queryPressed(GLFW_KEY_1)) {
		rs.showGBuffer = 1;
	}

	if (queryPressed(GLFW_KEY_2)) {
		rs.showGBuffer = 2;
	}

	if (queryPressed(GLFW_KEY_3)) {
		rs.showGBuffer = 3;
	}

	if (queryPressed(GLFW_KEY_4)) {
		rs.showGBuffer = 4;
	}

	if (queryPressed(GLFW_KEY_5)) {
		rs.showGBuffer = 5;
	}

	if (queryPressed(GLFW_KEY_6)) {
		rs.showGBuffer = 6;
	}
}

void Sponza::onMouseScroll(double xOffset, double yOffset) {
	camControl->onMouseScroll(xOffset, yOffset, getDeltaTime());
}

void Sponza::onMouseMove(double xPos, double yPos) {
	camControl->onMouseMove(xPos, yPos, getDeltaTime());
}

void Sponza::onWindowClose() {
	abort = 1;
}

Dar::FrameData &Sponza::getFrameData() {
	return frameData[renderer.getBackbufferIndex()];
}

bool Sponza::loadAssets() {
	if (!loadPipelines()) {
		return false;
	}

	// TODO: load the scene in binary format + run MikkTSpace on the tangents in a preprocess step.
	//       During runtime we should only read the bin scene and upload the data to the gpu.
	// MikkTSpace tangents give slightly better results than the tangents in the gltf file.
	//SceneLoaderError sceneLoadErr = loadScene("res\\scenes\\Sponza\\glTF\\Sponza.gltf", scene, sceneLoaderFlags_overrideGenTangents);
	// .. but the algorithm is too slow for run-time evaluation.
	SceneLoaderError sceneLoadErr = loadScene("res\\scenes\\Sponza\\glTF\\Sponza.gltf", scene, sceneLoaderFlags_none);
	
	if (sceneLoadErr != SceneLoaderError::Success) {
		LOG(Error, "Failed to load scene!");
		return false;
	}

	// TODO: hard-coded for debugging. Add lights through scene editor/use scene lights
	/*LightNode *lDir = new LightNode;
	lDir->type = LightType::Directional;
	lDir->direction = Vec3{ -1.f, -1.f, 0.f };
	lDir->diffuse   = Vec3{ .3f, .3f, .3f };
	lDir->ambient   = Vec3{ .1f, .1f, .1f };
	lDir->specular  = Vec3{ 1.f, 1.f, 1.f };
	scene.addNewLight(lDir);*/

	Dar::Random rand;
	for (int i = 0; i < 10; ++i) {
		const float x = rand.generateFlt(-200.f, 200.f);
		const float y = rand.generateFlt(0.f, 1.f);
		const float z = rand.generateFlt(-200.f, 200.f);
		const float r = rand.generateFlt(0.f, 1.f) * 1.f;
		const float g = rand.generateFlt(0.f, 1.f) * 1.f;
		const float b = rand.generateFlt(0.f, 1.f) * 1.f;

		LightNode *lPoint = new LightNode;
		lPoint->lightData.type = LightType::Point;
		lPoint->lightData.position = Vec3{ x, y, z };
		lPoint->lightData.diffuse = Vec3{ r, g, b };
		lPoint->lightData.ambient = Vec3{ .1f, .1f, .1f };
		lPoint->lightData.specular = Vec3{ 1.f, 0.f, 0.f };
		lPoint->lightData.attenuation = Vec3{ 1.f, 0.0014f, 0.000007f };
		scene.addNewLight(lPoint);
	}

	LightNode *lSpot = new LightNode;
	lSpot->lightData.type = LightType::Spot;
	lSpot->lightData.position = Vec3{ 0.f, 100.f, 0.f };
	lSpot->lightData.direction = Vec3{ 0.f, 0.f, 1.f };
	lSpot->lightData.diffuse  = Vec3{ .9f, .9f, .9f };
	lSpot->lightData.ambient  = Vec3{ .05f, .05f, .05f };
	lSpot->lightData.specular = Vec3{ 1.f, 1.f, 1.0f };
	lSpot->lightData.innerAngleCutoff = dmath::radians(7.f);
	lSpot->lightData.outerAngleCutoff = dmath::radians(40.f);
	scene.addNewLight(lSpot);

	Dar::Camera cam = Dar::Camera::perspectiveCamera(Vec3(0.f, -0.f, -0.f), 90.f, getWidth() / static_cast<float>(getHeight()), 0.1f, 10000.f);
	CameraNode *camNode = new CameraNode(std::move(cam));
	scene.addNewCamera(camNode);

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

	Dar::ResourceManager &resManager = Dar::getResourceManager();
	Dar::UploadHandle uploadHandle = resManager.beginNewUpload();

	if (!scene.uploadSceneData(uploadHandle)) {
		LOG(Error, "Failed to upload scene data!");
		return false;
	}

	if (!prepareVertexIndexBuffers(uploadHandle)) {
		LOG(Error, "Failed to prepare vertex and index buffers!");
		return false;
	}

	resManager.uploadBuffers();

	return true;
}

bool Sponza::updateRenderTargetViews() {
	auto getGBufferName = [](GBuffer type) -> WString {
		WString res;
		switch (type) {
		case GBuffer::Albedo:
			res = L"GBuffer::Albedo";
			break;
		case GBuffer::Normals:
			res = L"GBuffer::Normals";
			break;
		case GBuffer::MetallnessRoughnessOcclusion:
			res = L"GBuffer::MetallnessRoughnessOcclusion";
			break;
		case GBuffer::Position:
			res = L"GBuffer::Position";
			break;
		default:
			res = L"GBuffer::Unknown";
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

		gBufferRTs[i].init(rtvTextureDesc, Dar::FRAME_COUNT);
		gBufferRTs[i].setName(getGBufferName(static_cast<GBuffer>(i)));
	}

	WString lightPassRTName = L"LightPassRTV";
	Dar::TextureInitData rtvTextureDesc = {};
	rtvTextureDesc.width = width;
	rtvTextureDesc.height = height;
	rtvTextureDesc.format = DXGI_FORMAT_R8G8B8A8_UNORM;

	lightPassRT.init(rtvTextureDesc, Dar::FRAME_COUNT);
	lightPassRT.setName(lightPassRTName);

	return true;
}

bool Sponza::resizeDepthBuffer() {
	auto device = renderer.getDevice();
	return depthBuffer.init(device, width, height, DXGI_FORMAT_D32_FLOAT);
}

bool Sponza::loadPipelines() {
	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	CD3DX12_STATIC_SAMPLER_DESC sampler{ 0, D3D12_FILTER_MIN_MAG_MIP_LINEAR };
	
	Dar::RenderPassDesc deferredPassDesc = {};
	Dar::PipelineStateDesc deferredPSDesc = {};
	deferredPSDesc.shaderName = L"deferred";
	deferredPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	deferredPSDesc.inputLayouts = inputLayouts;
	deferredPSDesc.staticSamplerDesc = &sampler;
	deferredPSDesc.numInputLayouts = _countof(inputLayouts);
	deferredPSDesc.depthStencilBufferFormat = depthBuffer.getFormatAsDepthBuffer();
	deferredPSDesc.numConstantBufferViews = static_cast<UINT>(ConstantBufferView::Count);
	deferredPSDesc.numTextures = static_cast<UINT>(scene.getNumTextures());
	deferredPSDesc.numRenderTargets = static_cast<UINT>(GBuffer::Count);
	for (UINT i = 0; i < deferredPSDesc.numRenderTargets; ++i) {
		deferredPSDesc.renderTargetFormats[i] = gBufferFormats[i];
	}

	deferredPassDesc.setPipelineStateDesc(deferredPSDesc);
	for (int i = 0; i < static_cast<int>(GBuffer::Count); ++i) {
		deferredPassDesc.attach(Dar::RenderPassAttachment::renderTarget(&gBufferRTs[i]));
	}
	deferredPassDesc.attach(Dar::RenderPassAttachment::depthStencil(&depthBuffer, true));
	renderer.addRenderPass(deferredPassDesc);

	Dar::RenderPassDesc lightingPassDesc = {};
	Dar::PipelineStateDesc lightingPSDesc = {};
	lightingPSDesc.shaderName = L"lighting";
	lightingPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	lightingPSDesc.staticSamplerDesc = &sampler;
	lightingPSDesc.numConstantBufferViews = static_cast<unsigned int>(ConstantBufferView::Count);
	lightingPSDesc.numTextures = static_cast<UINT>(GBuffer::Count);
	lightingPSDesc.cullMode = D3D12_CULL_MODE_NONE;
	lightingPassDesc.setPipelineStateDesc(lightingPSDesc);
	lightingPassDesc.attach(Dar::RenderPassAttachment::renderTarget(&lightPassRT));
	renderer.addRenderPass(lightingPassDesc);

	Dar::RenderPassDesc postPassDesc = {};
	Dar::PipelineStateDesc postPSDesc = {};
	postPSDesc.shaderName = L"post";
	postPSDesc.shadersMask = Dar::shaderInfoFlags_useVertex;
	postPSDesc.staticSamplerDesc = &sampler;
	postPSDesc.numConstantBufferViews = static_cast<unsigned int>(ConstantBufferView::Count);
	postPSDesc.numTextures = 1;
	postPSDesc.cullMode = D3D12_CULL_MODE_NONE;
	postPassDesc.setPipelineStateDesc(postPSDesc);
	postPassDesc.attach(Dar::RenderPassAttachment::renderTargetBackbuffer());
	renderer.addRenderPass(postPassDesc);

	renderer.compilePipeline();

	return true;
}

bool Sponza::prepareVertexIndexBuffers(Dar::UploadHandle uploadHandle) {
	Dar::VertexIndexBufferDesc vertexDesc = {};
	vertexDesc.data = scene.getVertexBuffer();
	vertexDesc.size = scene.getVertexBufferSize();
	vertexDesc.name = L"VertexBuffer";
	vertexDesc.vertexBufferStride = sizeof(Vertex);
	if (!vertexBuffer.init(vertexDesc, uploadHandle)) {
		return false;
	}

	Dar::VertexIndexBufferDesc indexDesc = {};
	indexDesc.data = scene.getIndexBuffer();
	indexDesc.size = scene.getIndexBufferSize();
	indexDesc.name = L"IndexBuffer";
	indexDesc.indexBufferFormat = DXGI_FORMAT_R32_UINT;
	if (!indexBuffer.init(indexDesc, uploadHandle)) {
		return false;
	}

	return true;
}
