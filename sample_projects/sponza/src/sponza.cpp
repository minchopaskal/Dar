#include "sponza.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "d3d12_app.h"
#include "d3d12_asset_manager.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_resource_manager.h"
#include "d3d12_utils.h"
#include "scene_loader.h"

#include "gpu_cpu_common.hlsli"

#include "GLFW/glfw3.h" // keyboard input

#include "imgui.h"

Sponza::Sponza(const UINT w, const UINT h, const String &windowTitle) :
	D3D12App(w, h, windowTitle.c_str()),
	sceneDataHandle{ INVALID_RESOURCE_HANDLE, INVALID_RESOURCE_HANDLE },
	viewport{ 0.f, 0.f, static_cast<float>(w), static_cast<float>(h), 0.f, 1.f },
	scissorRect{ 0, 0, LONG_MAX, LONG_MAX }, // always render on the entire screen
	aspectRatio(static_cast<float>(w) / static_cast<float>(h)),
	fenceValues{ 0 },
	scene(device),
	camControl(nullptr),
	fpsModeControl(nullptr, 200.f),
	editModeControl(nullptr, 200.f),
	fps(0.0),
	totalTime(0.0),
	deltaTime(0.0),
	showGBuffer(0),
	editMode(true),
	withNormalMapping(true)
{
	camControl = editMode ? &editModeControl : &fpsModeControl;
	gBufferRTVTextureHandles.fill(INVALID_RESOURCE_HANDLE);
}

void setGLFWCursorHiddenState(GLFWwindow *window, bool show) {
	glfwSetInputMode(window, GLFW_CURSOR, show ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

int Sponza::initImpl() {
	setUseImGui();

	fpsModeControl.window = editModeControl.window = getGLFWWindow();

	setGLFWCursorHiddenState(getGLFWWindow(), editMode == true);

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_6 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.6!"
	);

	RETURN_ERROR_IF(shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_6, false, "Shader model 6.6 not supported!");

	/* Create a descriptor heap for RTVs */
	deferredRTVHeap.init(
		device.Get(),
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		static_cast<int>(GBuffer::Count) * frameCount, /*numDescriptors*/
		true /*shaderVisible*/
	);
	
	lightPassRTVHeap.init(
		device.Get(),
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		frameCount, /*numDescriptors*/
		true /*shaderVisible*/
	);

	postPassRTVHeap.init(
		device.Get(),
		D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		frameCount, /*numDescriptors*/
		false /*shaderVisible*/
	);

	if (!resizeDepthBuffer()) {
		return false;
	}

	if (!updateRenderTargetViews()) {
		return false;
	}

	return true;
}

void Sponza::deinit() {
	flush();
	Super::deinit();
}

void Sponza::update() {
	timeIt();

	camControl->processKeyboardInput(this, deltaTime);

	const Camera &cam = camControl->getCamera();

	// Update VP matrices
	Mat4 viewMat = cam.getViewMatrix();
	Mat4 projectionMat = cam.getProjectionMatrix();

	SceneData sceneData = {};

	sceneData.viewProjection = projectionMat * viewMat;
	sceneData.cameraPosition = Vec4{ cam.getPos(), 1.f };
	sceneData.cameraDir = Vec4{ dmath::normalized(cam.getCameraZ()), 1.f };
	sceneData.numLights = static_cast<int>(scene.getNumLights());
	sceneData.showGBuffer = showGBuffer;
	sceneData.width = width;
	sceneData.height = height;
	sceneData.withNormalMapping = withNormalMapping;

	/// Initialize the MVP constant buffer resource if needed
	if (sceneDataHandle[frameIndex] == INVALID_RESOURCE_HANDLE) {
		wchar_t frameMVPName[32] = L"";
		swprintf(frameMVPName, 32, L"SceneData[%d]", frameIndex);

		ResourceInitData resData(ResourceType::DataBuffer);
		resData.size = sizeof(SceneData);
		resData.name = frameMVPName;
		sceneDataHandle[frameIndex] = resManager->createBuffer(resData);
	}

	UploadHandle uploadHandle = resManager->beginNewUpload();
	resManager->uploadBufferData(uploadHandle, sceneDataHandle[frameIndex], reinterpret_cast<void*>(&sceneData), sizeof(SceneData));
	resManager->uploadBuffers();
}

void Sponza::render() {
	commandQueueDirect.addCommandListForExecution(populateCommandList());
	fenceValues[frameIndex] = commandQueueDirect.executeCommandLists();

	const UINT syncInterval = vSyncEnabled ? 1 : 0;
	const UINT presentFlags = allowTearing && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), ,"Failed to execute command list!");
	
	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// wait for the next frame's buffer
	commandQueueDirect.waitForFenceValue(fenceValues[frameIndex]);
}

void Sponza::drawUI() {
	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);

	const Camera &cam = camControl->getCamera();

	ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("FPS: %.2f", fps);
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
	ImGui::End();

	ImGui::Begin("General controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
		ImGui::Text("[`] - Show Rendered image");
		ImGui::Text("[1-4] - Show G-Buffers");
		ImGui::Text("[m] - Switch between FPS/edit modes");
		ImGui::Text("[o] - Switch between perspective/orthographic projection");
		ImGui::Text("[f] - Toggle fullscreen mode");
		ImGui::Text("[v] - Toggle V-Sync mode");
	ImGui::End();

	camControl->onDrawUI();

	if (editMode) {
		ImGui::Begin("Edit mode", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::ListBox("G-Buffer", &showGBuffer, gBufferLabels, 5);
			ImGui::Checkbox("With normal mapping", &withNormalMapping);
			ImGui::Checkbox("V-Sync", &vSyncEnabled);
		ImGui::End();
	}
}

void Sponza::onResize(const unsigned int w, const unsigned int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1u, w);
	this->height = std::max(1u, h);
	viewport = { 0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f };
	aspectRatio = static_cast<float>(width) / static_cast<float>(height);

	flush();

	for (unsigned int i = 0; i < frameCount; ++i) {
		backBuffers[i].Reset();
		// It's important to deregister an outside resource if you want it deallocated
		// since the ResourceManager keeps a ref if it was registered with it.
		resManager->deregisterResource(backBuffersHandles[i]);
	}

	DXGI_SWAP_CHAIN_DESC scDesc = { };
	RETURN_ON_ERROR(
		swapChain->GetDesc(&scDesc), ,
		"Failed to retrieve swap chain's description"
	);
	RETURN_ON_ERROR(
		swapChain->ResizeBuffers(
			frameCount,
			this->width,
			this->height,
			scDesc.BufferDesc.Format,
			scDesc.Flags
		), ,
		"Failed to resize swap chain buffer"
	);

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	updateRenderTargetViews();

	resizeDepthBuffer();
}

void Sponza::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		vSyncEnabled = !vSyncEnabled;
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
		showGBuffer = 0;
	}

	if (queryPressed(GLFW_KEY_1)) {
		showGBuffer = 1;
	}

	if (queryPressed(GLFW_KEY_2)) {
		showGBuffer = 2;
	}

	if (queryPressed(GLFW_KEY_3)) {
		showGBuffer = 3;
	}

	if (queryPressed(GLFW_KEY_4)) {
		showGBuffer = 4;
	}
}

void Sponza::onMouseScroll(double xOffset, double yOffset) {
	camControl->onMouseScroll(xOffset, yOffset, deltaTime);
}

void Sponza::onMouseMove(double xPos, double yPos) {
	camControl->onMouseMove(xPos, yPos, deltaTime);
}

bool Sponza::loadAssets() {
	if (!loadPipelines()) {
		return false;
	}

	// MikkTSpace tangents give slightly better results than the tangents in the gltf file.
	SceneLoaderError sceneLoadErr = loadScene("res\\scenes\\Sponza\\glTF\\Sponza.gltf", scene, sceneLoaderFlags_none);
	if (sceneLoadErr != SceneLoaderError::Success) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to load scene!");
		return false;
	}

	// TODO: hard-coded for debugging. Add lights through scene editor/use scene lights
	LightNode *lDir = new LightNode;
	lDir->type = LightType::Directional;
	lDir->direction = Vec3{ -1.f, -1.f, 0.f };
	lDir->diffuse   = Vec3{ .3f, .3f, .3f };
	lDir->ambient   = Vec3{ .1f, .1f, .1f };
	lDir->specular  = Vec3{ 1.f, 1.f, 1.f };
	scene.addNewLight(lDir);

	LightNode *lPoint = new LightNode;
	lPoint->type = LightType::Point;
	lPoint->position = Vec3{ -400.f, 200.f, 0.f };
	lPoint->diffuse = Vec3{ .7f, .0f, .0f };
	lPoint->ambient = Vec3{ .1f, .1f, .1f };
	lPoint->specular = Vec3{ 1.f, 0.f, 0.f };
	lPoint->attenuation = Vec3{ 1.f, 0.0014f, 0.000007f };
	scene.addNewLight(lPoint);

	LightNode *lSpot = new LightNode;
	lSpot->type = LightType::Spot;
	lSpot->position = Vec3{ 0.f, 100.f, 0.f };
	lSpot->direction = Vec3{ 0.f, 0.f, 1.f };
	lSpot->diffuse  = Vec3{ .9f, .9f, .9f };
	lSpot->ambient  = Vec3{ .05f, .05f, .05f };
	lSpot->specular = Vec3{ 1.f, 1.f, 1.0f };
	lSpot->innerAngleCutoff = dmath::radians(35.5f);
	lSpot->outerAngleCutoff = dmath::radians(40.f);
	scene.addNewLight(lSpot);

	Camera cam = Camera::perspectiveCamera(Vec3(0.f, 0.f, -1.f), 90.f, getWidth() / static_cast<float>(getHeight()), 0.1f, 10000.f);
	CameraNode *camNode = new CameraNode(std::move(cam));
	scene.addNewCamera(camNode);

	bool setCamRes = scene.setCameraForCameraController(fpsModeControl);
	if (!setCamRes) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to set FPS camera controller!");
		return false;
	}
	setCamRes = scene.setCameraForCameraController(editModeControl);
	if (!setCamRes) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to set Edit mode camera controller!");
		return false;
	}

	ResourceManager &resManager = getResourceManager();
	UploadHandle uploadHandle = resManager.beginNewUpload();

	if (!scene.uploadSceneData(uploadHandle)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to upload scene data!");
		return false;
	}

	if (!prepareVertexIndexBuffers(uploadHandle)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to prepare vertex and index buffers!");
		return false;
	}

	resManager.uploadBuffers();

	return true;
}

CommandList Sponza::populateCommandList() {
	CommandList commandList = commandQueueDirect.getCommandList();

	if (!commandList.isValid()) {
		return commandList;
	}

	WString commandListName = L"CommandList[" + std::to_wstring(frameIndex) + L"]";
	commandList->SetName(commandListName.c_str());
	
	populateDeferredPassCommands(commandList);
	populateLightPassCommands(commandList);
	populateForwardPassCommands(commandList);
	populatePostPassCommands(commandList);

	return commandList;
}

void Sponza::populateDeferredPassCommands(CommandList& commandList) {
	commandList->SetPipelineState(deferredPassPipelineState.getPipelineState());

	commandList.transition(scene.materialsHandle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	const int numTextures = static_cast<int>(scene.getNumTextures());
	for (int i = 0; i < numTextures; ++i) {
		commandList.transition(scene.textureHandles[i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	}

	if (!deferredPassSRVHeap[frameIndex] || scene.hadChangesSinceLastCheck()) {
		/* Create shader resource view heap which will store the handles to the textures */
		deferredPassSRVHeap[frameIndex].init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
			numTextures + 1, // +1 for the materials buffer
			true /*shaderVisible*/
		);
	}

	// Reset the handles in the heap since they may have been changed since the last frame.
	// F.e added/removed textures.
	deferredPassSRVHeap[frameIndex].reset();

	// Create SRV for the materials
	deferredPassSRVHeap[frameIndex].addBufferSRV(scene.materialsHandle.get(), static_cast<int>(scene.getNumMaterials()), sizeof(GPUMaterial));

	// Create SRVs for the textures so we can read them bindlessly in the shader
	for (int i = 0; i < numTextures; ++i) {
		deferredPassSRVHeap[frameIndex].addTexture2DSRV(
			scene.textureHandles[i].get(),
			scene.getTexture(i).format == TextureFormat::RGBA_8BIT ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN
		);
	}

	commandList->SetDescriptorHeaps(1, deferredPassSRVHeap[frameIndex].getAddressOf());

	commandList->SetGraphicsRootSignature(deferredPassPipelineState.getRootSignature());

	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	const int gBufferCount = static_cast<int>(GBuffer::Count);
	for (int i = 0; i < gBufferCount; ++i) {
		commandList.transition(gBufferRTVTextureHandles[frameIndex * gBufferCount + i], D3D12_RESOURCE_STATE_RENDER_TARGET);
	}

	const D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = deferredRTVHeap.getCPUHandle(static_cast<int>(frameIndex) * gBufferCount);
	const D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = depthBuffer.getCPUHandle();

	constexpr float clearColor[] = { 0.f, 0.f, 0.f, 1.f };
	for (int i = 0; i < gBufferCount; ++i) {
		constexpr float clearColor2[] = { 0.f, 0.5f, 0.8f, 1.f };
		commandList->ClearRenderTargetView(deferredRTVHeap.getCPUHandle(static_cast<int>(frameIndex) * gBufferCount + i), i == 0 ? clearColor2 : clearColor, 0, nullptr);
	}
	commandList.transition(depthBuffer.getBufferHandle(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
	commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);

	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	commandList.transition(vertexBuffer.bufferHandle, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
	commandList.transition(indexBuffer.bufferHandle, D3D12_RESOURCE_STATE_INDEX_BUFFER);
	commandList.transition(sceneDataHandle[frameIndex], D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

	commandList->IASetVertexBuffers(0, 1, &vertexBuffer.bufferView);
	commandList->IASetIndexBuffer(&indexBuffer.bufferView);
	commandList->OMSetRenderTargets(gBufferCount, &rtvHandle, TRUE, &dsvHandle);
	commandList.setMVPBuffer(sceneDataHandle[frameIndex]);

	scene.draw(commandList);
}

void Sponza::populateLightPassCommands(CommandList& commandList) {
	commandList->SetPipelineState(lightPassPipelineState.getPipelineState());

	commandList.transition(scene.lightsHandle, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	commandList.transition(depthBuffer.getBufferHandle(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

	const int gBufferCount = static_cast<int>(GBuffer::Count);
	
	for (int i = 0; i < gBufferCount; ++i) {
		commandList.transition(gBufferRTVTextureHandles[frameIndex * gBufferCount + i], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	}

	if (!lightPassSRVHeap[frameIndex]) {
		lightPassSRVHeap[frameIndex].init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
			gBufferCount + 2, // Gbuffer textures + lights buffer + depth buffer
			true /*shaderVisible*/
		);
	}

	// Recreate the srv heap since the lights may have been changed
	// (thus scene.lightsHandle update is needed) or in case of
	// resizing the the RTV handles will be diffrent and will also need an update.
	lightPassSRVHeap[frameIndex].reset();

	// Create SRV for the lights
	lightPassSRVHeap[frameIndex].addBufferSRV(scene.lightsHandle.get(), static_cast<int>(scene.getNumLights()), sizeof(GPULight));

	// ... and for the depth buffer
	lightPassSRVHeap[frameIndex].addTexture2DSRV(depthBuffer.getBufferResource(), depthBuffer.getFormatAsTexture());

	// Create SRVs for the textures so we can read them bindlessly in the shader
	for (int i = 0; i < gBufferCount; ++i) {
		lightPassSRVHeap[frameIndex].addTexture2DSRV(gBufferRTVTextureHandles[frameIndex * gBufferCount + i].get(), gBufferFormats[i]);
	}

	commandList->SetDescriptorHeaps(1, lightPassSRVHeap[frameIndex].getAddressOf());

	commandList->SetGraphicsRootSignature(lightPassPipelineState.getRootSignature());

	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	commandList.transition(lightPassRTVTextureHandles[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET);

	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = lightPassRTVHeap.getCPUHandle(static_cast<int>(frameIndex));

	constexpr float clearColor[] = { 0.f, 0.3f, 0.7f, 1.f };
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	commandList.setMVPBuffer(sceneDataHandle[frameIndex]);

	commandList->DrawIndexedInstanced(3, 1, 0, 0, 0);
}

void Sponza::populateForwardPassCommands(CommandList& cmdList) {}

void Sponza::populatePostPassCommands(CommandList&commandList) {
	commandList->SetPipelineState(postPassPipelineState.getPipelineState());

	commandList.transition(lightPassRTVTextureHandles[frameIndex], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

	if (!postPassSRVHeap[frameIndex]) {
		postPassSRVHeap[frameIndex].init(
			device.Get(),
			D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
			2, // light pass result + depth buffer
			true /*shaderVisible*/
		);
	}

	// Recreate the srv heap since after resizing the light pass rtv texture handle may need an update.
	postPassSRVHeap[frameIndex].reset();
	postPassSRVHeap[frameIndex].addTexture2DSRV(lightPassRTVTextureHandles[frameIndex].get(), DXGI_FORMAT_R8G8B8A8_UNORM);
	postPassSRVHeap[frameIndex].addTexture2DSRV(depthBuffer.getBufferResource(), depthBuffer.getFormatAsTexture());

	commandList->SetDescriptorHeaps(1, postPassSRVHeap[frameIndex].getAddressOf());

	commandList->SetGraphicsRootSignature(postPassPipelineState.getRootSignature());

	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	ID3D12DebugCommandList1 *dcl = nullptr;
	commandList->QueryInterface<ID3D12DebugCommandList1>(&dcl);
	if (dcl) {
		dcl->AssertResourceState(backBuffers[frameIndex].Get(), 0, D3D12_RESOURCE_STATE_PRESENT);
	}

	//commandList.transition(backBuffersHandles[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET);
	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET));

	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = postPassRTVHeap.getCPUHandle(static_cast<int>(frameIndex));

	constexpr float clearColor[] = { 0.f, 0.f, 0.f, 1.f };
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	commandList.setMVPBuffer(sceneDataHandle[frameIndex]);

	commandList->DrawIndexedInstanced(3, 1, 0, 0, 0);

	renderUI(commandList, rtvHandle);

	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON));
	commandList.transition(backBuffersHandles[frameIndex], D3D12_RESOURCE_STATE_PRESENT);
}

bool Sponza::updateRenderTargetViews() {
	auto getGBufferName = [](wchar_t backBufferName[32], GBuffer type, int frameIndex) -> wchar_t* {
		int offset = 0;
		switch (type) {
		case GBuffer::Diffuse:
			offset = swprintf(backBufferName, 32, L"GBuffer::Diffuse");
			break;
		case GBuffer::Normals:
			offset = swprintf(backBufferName, 32, L"GBuffer::Normals");
			break;
		case GBuffer::Specular:
			offset = swprintf(backBufferName, 32, L"GBuffer::Specular");
			break;
		case GBuffer::Position:
			offset = swprintf(backBufferName, 32, L"GBuffer::Position");
			break;
		default:
			offset = swprintf(backBufferName, 32, L"GBuffer::Unknown");
			break;
		}

		swprintf(backBufferName + offset, 32 - offset, L"[%d]", frameIndex);
		return backBufferName;
	};

	postPassRTVHeap.reset();
	for (UINT i = 0; i < frameCount; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);
		postPassRTVHeap.addRTV(backBuffers[i].Get(), nullptr);

		// Register the back buffer's resources manually since the resource manager doesn't own them, the swap chain does.
#ifdef D3D12_DEBUG
		backBuffersHandles[i] = resManager->registerResource(backBuffers[i].Get(), 1, D3D12_RESOURCE_STATE_RENDER_TARGET, ResourceType::RenderTargetBuffer);
#else
		backBuffersHandles[i] = resManager->registerResource(backBuffers[i].Get(), 1, D3D12_RESOURCE_STATE_PRESENT);
#endif

		wchar_t backBufferName[32];
		swprintf(backBufferName, 32, L"BackBuffer[%u]", i);
		backBuffers[i]->SetName(backBufferName);
	}

	const int gBufferCount = static_cast<UINT>(GBuffer::Count);
	deferredRTVHeap.reset();
	lightPassRTVHeap.reset();
	for (UINT i = 0; i < frameCount; ++i) {
		for (int j = 0; j < gBufferCount; ++j) {
			ResourceHandle& rtvTexture = gBufferRTVTextureHandles[i * gBufferCount + j];
			if (rtvTexture != INVALID_RESOURCE_HANDLE) {
				resManager->deregisterResource(rtvTexture);
			}

			wchar_t rtvTextureName[32];
			ResourceInitData rtvTextureDesc(ResourceType::RenderTargetBuffer);
			rtvTextureDesc.textureData.width = width;
			rtvTextureDesc.textureData.height = height;
			rtvTextureDesc.textureData.format = gBufferFormats[j];
			rtvTextureDesc.name = getGBufferName(rtvTextureName, static_cast<GBuffer>(j), i);
			rtvTexture = resManager->createBuffer(rtvTextureDesc);

			deferredRTVHeap.addRTV(rtvTexture.get(), nullptr);
		}

		// Light pass RTV preparation
		{
			ResourceHandle &rtvTexture = lightPassRTVTextureHandles[i];
			if (rtvTexture != INVALID_RESOURCE_HANDLE) {
				resManager->deregisterResource(rtvTexture);
			}

			wchar_t rtvTextureName[32];
			swprintf(rtvTextureName, L"LightPassRTV[%d]", i);

			ResourceInitData rtvTextureDesc(ResourceType::RenderTargetBuffer);
			rtvTextureDesc.textureData.width = width;
			rtvTextureDesc.textureData.height = height;
			rtvTextureDesc.textureData.format = DXGI_FORMAT_R8G8B8A8_UNORM;
			rtvTextureDesc.name = rtvTextureName;
			rtvTexture = resManager->createBuffer(rtvTextureDesc);

			lightPassRTVHeap.addRTV(rtvTexture.get(), nullptr);
		}
	}

	return true;
}

bool Sponza::resizeDepthBuffer() {
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
	PipelineStateDesc deferredPSDesc = {};
	deferredPSDesc.shaderName = L"deferred";
	deferredPSDesc.shadersMask = shaderInfoFlags_useVertex;
	deferredPSDesc.inputLayouts = inputLayouts;
	deferredPSDesc.staticSamplerDesc = &sampler;
	deferredPSDesc.numInputLayouts = _countof(inputLayouts);
	deferredPSDesc.depthStencilBufferFormat = depthBuffer.getFormatAsDepthBuffer();
	deferredPSDesc.numConstantBufferViews = static_cast<UINT>(ConstantBufferView::Count);
	deferredPSDesc.numTextures = static_cast<UINT>(scene.getNumTextures());
	deferredPSDesc.maxVersion = rootSignatureFeatureData.HighestVersion;
	deferredPSDesc.numRenderTargets = static_cast<UINT>(GBuffer::Count);
	for (UINT i = 0; i < deferredPSDesc.numRenderTargets; ++i) {
		deferredPSDesc.renderTargetFormats[i] = gBufferFormats[i];
	}
	if (!deferredPassPipelineState.init(device, deferredPSDesc)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to initialize deferred pass pipeline!");
		return false;
	}

	PipelineStateDesc lightingPSDesc = {};
	lightingPSDesc.shaderName = L"lighting";
	lightingPSDesc.shadersMask = shaderInfoFlags_useVertex;
	lightingPSDesc.staticSamplerDesc = &sampler;
	lightingPSDesc.numConstantBufferViews = static_cast<unsigned int>(ConstantBufferView::Count);
	lightingPSDesc.numTextures = static_cast<UINT>(GBuffer::Count);
	lightingPSDesc.maxVersion = rootSignatureFeatureData.HighestVersion;
	if (!lightPassPipelineState.init(device, lightingPSDesc)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to initialize light pass pipeline!");
		return false;
	}

	PipelineStateDesc postPSDesc = {};
	postPSDesc.shaderName = L"post";
	postPSDesc.shadersMask = shaderInfoFlags_useVertex;
	postPSDesc.staticSamplerDesc = &sampler;
	postPSDesc.numConstantBufferViews = static_cast<unsigned int>(ConstantBufferView::Count);
	postPSDesc.numTextures = 1;
	postPSDesc.maxVersion = rootSignatureFeatureData.HighestVersion;
	if (!postPassPipelineState.init(device, postPSDesc)) {
		D3D12::Logger::log(D3D12::LogLevel::Error, "Failed to initialize post pass pipeline!");
		return false;
	}

	return true;
}

bool Sponza::prepareVertexIndexBuffers(UploadHandle uploadHandle) {
	VertexIndexBufferDesc vertexDesc = {};
	vertexDesc.data = scene.getVertexBuffer();
	vertexDesc.size = scene.getVertexBufferSize();
	vertexDesc.name = L"VertexBuffer";
	vertexDesc.vertexBufferStride = sizeof(Vertex);
	if (!vertexBuffer.init(vertexDesc, uploadHandle)) {
		return false;
	}

	VertexIndexBufferDesc indexDesc = {};
	indexDesc.data = scene.getIndexBuffer();
	indexDesc.size = scene.getIndexBufferSize();
	indexDesc.name = L"IndexBuffer";
	indexDesc.indexBufferFormat = DXGI_FORMAT_R32_UINT;
	if (!indexBuffer.init(indexDesc, uploadHandle)) {
		return false;
	}

	return true;
}

void Sponza::timeIt() {
	using std::chrono::duration_cast;
	using Hrc = std::chrono::high_resolution_clock;

	static constexpr double seconds_in_nanosecond = 1e-9;
	static UINT64 frameCount = 0;
	static double elapsedTime = 0.0;
	static Hrc::time_point t0 = Hrc::now();

	const Hrc::time_point t1 = Hrc::now();
	deltaTime = static_cast<double>((t1 - t0).count()) * seconds_in_nanosecond;
	elapsedTime += deltaTime;
	totalTime += deltaTime;
	
	++frameCount;
	t0 = t1;

	if (elapsedTime > 1.0) {
		fps = static_cast<double>(frameCount) / elapsedTime;

#if defined(D3D12_DEBUG)
		char buffer[512];
		sprintf_s(buffer, "FPS: %.2f\n", fps);
		OutputDebugString(buffer);
#endif // defined(D3D12_DEBUG)

		frameCount = 0;
		elapsedTime = 0.0;
	}
}
