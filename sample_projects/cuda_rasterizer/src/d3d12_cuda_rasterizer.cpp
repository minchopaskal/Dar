#include "d3d12_cuda_rasterizer.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "d3d12_app.h"
#include "d3d12_resource_manager.h"
#include "d3d12_asset_manager.h"
#include "d3d12_utils.h"

#include "cuda_manager.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

#include "imgui.h"

CudaRasterizer::CudaRasterizer(UINT width, UINT height, const String &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	rtvHeapHandleIncrementSize(0),
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, LONG_MAX, LONG_MAX }, // always render on the entire screen
	aspectRatio(width / float(height)),
	cudaRenderTargetHost{ nullptr },
	fenceValues{ 0 },
	previousFrameIndex(0),
	cudaManager(&getCUDAManager()),
	FOV(45.0),
	fps(0.0),
	totalTime(0.0)
{ }

int CudaRasterizer::init() {
	if (!D3D12App::init()) {
		return false;
	}

	setUseImGui();

	/* Create a descriptor heap for RTVs */
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { };
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.NumDescriptors = frameCount;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	rtvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)),
		"Failed to create RTV descriptor heap!"
	);

	rtvHeapHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	// Srv heap storing the texture used as CUDA render target
	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.NumDescriptors = 1;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	srvHeapDesc.NodeMask = 0;
	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap)),
		"Failed to create DSV descriptor heap!"
	);

	if (!updateRenderTargetViews()) {
		return false;
	}

	return loadAssets();
}

int CudaRasterizer::loadScene(const String &name) {
	Mesh *mesh = new Mesh("res\\obj\\head.obj", "Main");

	scene = mesh;
}

void CudaRasterizer::deinit() {
	flush();
	deinitCUDAData();
	delete scene;
	Super::deinit();
}

void CudaRasterizer::update() {
	timeIt();

	const CUDADevice &device = cudaManager->getDevices()[0];
	device.use();

	setClearColor(Vec4{ 0.1f, 0.3f, 1.f, 1.f });
	clearRenderTarget();

	clearDepthBuffer();

	// Render the image with CUDA
	scene->draw(*this);

	CUstream stream = device.getDefaultStream(CUDADefaultStreamsEnumeration::Execution);
	cuStreamSynchronize(stream);

	cudaRenderTargetDevice.download(cudaRenderTargetHost);
}

void CudaRasterizer::render() {
	// Upload the rendered image as a D3D12 texture
	UploadHandle handle = resManager->beginNewUpload();
	D3D12_SUBRESOURCE_DATA subresData;
	subresData.pData = cudaRenderTargetHost;
	subresData.RowPitch = SizeType(width) * numComps * sizeof(float);
	subresData.SlicePitch = subresData.RowPitch * height;
	resManager->uploadTextureData(handle, dx12RenderTargetHandle, &subresData, 1, 0);
	resManager->uploadBuffers();

	// Render a screen-quad with the texture
	CommandList cmdList = populateCommandList();
	commandQueueDirect.addCommandListForExecution(std::move(cmdList));

	fenceValues[frameIndex] = commandQueueDirect.executeCommandLists();

	UINT syncInterval = vSyncEnabled ? 1 : 0;
	UINT presentFlags = allowTearing && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), , "Failed to execute command list!");

	previousFrameIndex = frameIndex;
	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// wait for the next frame. This waits for the fence signaled by the DirectX command queue.
	commandQueueDirect.waitForFenceValue(fenceValues[frameIndex]);
}

void CudaRasterizer::onResize(int w, int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1, w);
	this->height = std::max(1, h);
	viewport = { 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) };
	aspectRatio = width / float(height);

	flush();

	for (int i = 0; i < frameCount; ++i) {
		backBuffers[i].Reset();
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

	initCUDAData();
}

void CudaRasterizer::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		vSyncEnabled = !vSyncEnabled;
	}
}

void CudaRasterizer::onMouseScroll(double xOffset, double yOffset) {
	static const double speed = 500.f;
	FOV -= float(speed * deltaTime * yOffset);
	FOV = dmath::min(dmath::max(30.f, FOV), 120.f);
}

void CudaRasterizer::drawUI() {
	Super::drawUI();

	ImGui::Begin("FPS Counter");
	ImGui::Text("FPS: %.2f", fps);
	ImGui::End();
}

void CudaRasterizer::setUseDepthBuffer(bool useDepthBuffer) {
	const CUDADevice &device = cudaManager->getDevices()[0];

	device.uploadConstantParam<bool>(&useDepthBuffer, "useDepthBuffer");

	CUDAMemHandle depthBufferHandle = NULL;
	if (useDepthBuffer) {
		if (depthBuffer.getSize() == SizeType(width) * height * sizeof(float)) {
			depthBufferHandle = depthBuffer.handle();
		} else {
			depthBuffer.initialize(SizeType(width) * height * sizeof(float));
			depthBufferHandle = depthBuffer.handle();
			clearDepthBuffer();
		}
	}

	device.uploadConstantParam(&depthBufferHandle, "depthBuffer");
}

void CudaRasterizer::setVertexBuffer(const Vertex *buffer, SizeType verticesCount) {
	const CUDADevice &device = cudaManager->getDevices()[0];

	vertexBuffer.initialize(verticesCount * sizeof(Vertex));
	vertexBuffer.upload(buffer);

	CUDAMemHandle vertexBufferHandle = vertexBuffer.handle();

	device.uploadConstantParam(&vertexBufferHandle, "vertexBuffer");
}

void CudaRasterizer::setIndexBuffer(const unsigned int *buffer, SizeType indicesCount) {
	const CUDADevice &device = cudaManager->getDevices()[0];

	indexBuffer.initialize(indicesCount * sizeof(unsigned int));
	indexBuffer.upload(reinterpret_cast<const void*>(buffer));

	CUDAMemHandle indexBufferHandle = indexBuffer.handle();

	device.uploadConstantParam(&indexBufferHandle, "indexBuffer");
}

void CudaRasterizer::setUAVBuffer(const void *buffer, SizeType size, int index) {
	const CUDADevice &device = cudaManager->getDevices()[0];

	uavBuffers[index].initialize(size);
	uavBuffers[index].upload(reinterpret_cast<const void*>(buffer));

	CUDAMemHandle uavBufferHandle = uavBuffers[index].handle();

	device.uploadConstantParam(&uavBufferHandle, "resources", SizeType(index));
}

void CudaRasterizer::setVertexShader(const String &name) {
	// todo
}

void CudaRasterizer::setPixelShader(const String &name) {
	// todo
}

bool CudaRasterizer::drawIndexed(const unsigned int numPrimitives) {
	const CUDADevice &device = cudaManager->getDevices()[0];

	// Vertex shading
	// primitive assembly
	// eventually culling
	// vertex attributes interpolation
	// pixel shading


	// Vertex processing
	CUDAFunction drawIndexedFunc(device.getModule(), "drawIndexed");
	drawIndexedFunc.addParams(
		numPrimitives,
		width,
		height
	);
	
	CUDAError err = drawIndexedFunc.launchSync(numPrimitives, device.getDefaultStream(CUDADefaultStreamsEnumeration::Execution));

	// call shadePrimitive once for each prim

	return !err.hasError();
}

void CudaRasterizer::setClearColor(Vec4 color) {
	float colorHost[4];
	for (int i = 0; i < numComps; ++i) {
		colorHost[i] = color.data[i];
	}

	const CUDADevice &device = cudaManager->getDevices()[0];
	device.use();

	device.uploadConstantArray(colorHost, 4, "clearColor");
}

void CudaRasterizer::setCulling(CudaRasterizerCullType cullType) {
	const CUDADevice &device = cudaManager->getDevices()[0];
	device.use();

	device.uploadConstantParam(&cullType, "cullType");
}

void CudaRasterizer::clearRenderTarget() {
	const CUDADevice &device = cudaManager->getDevices()[0];
	device.use();
	CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	CUDAError err = handleCUDAError(cuMemsetD8(rt, 0, cudaRenderTargetDevice.getSize()));
	if (err.hasError()) {
		// TODO: handle error
		return;
	}

	CUDAFunction blankKernel(device.getModule(), "blank");
	blankKernel.addParams(
		rt,
		width,
		height
	);
	
	blankKernel.launchSync(width * height, device.getDefaultStream(CUDADefaultStreamsEnumeration::Execution));
}

void CudaRasterizer::clearDepthBuffer() {
	if (depthBuffer.getSize() == 0) {
		return;
	}

	const float floatOne = 1.f;
	const unsigned int floatOneBits = *reinterpret_cast< const unsigned int*>(&floatOne);

	CUDAMemHandle depthBufferHandle = depthBuffer.handle();
	cuMemsetD32(depthBufferHandle, floatOneBits, SizeType(width) * height);
}

int CudaRasterizer::loadAssets() {
	// Create the pipeline state
	PipelineStateDesc desc;
	desc.staticSamplerDesc = &CD3DX12_STATIC_SAMPLER_DESC(0);
	desc.shaderName = L"screen_quad";
	desc.shadersMask = sif_useVertex;
	desc.numTextures = 1;
	pipelineState.init(device, desc);

	// Create the texture resources that CUDA will use as render targets
	initCUDAData();

	return true;
}

CommandList CudaRasterizer::populateCommandList() {
	CommandList commandList = commandQueueDirect.getCommandList();

	// "Swap" the render targets
	D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = srvHeap->GetCPUDescriptorHandleForHeapStart();
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.Texture2D.MipLevels = 1;
	device->CreateShaderResourceView(dx12RenderTargetHandle.get(), &srvDesc, srvHandle);

	commandList->SetPipelineState(pipelineState.getPipelineState());
	commandList->SetGraphicsRootSignature(pipelineState.getRootSignature());

	commandList->SetDescriptorHeaps(1, srvHeap.GetAddressOf());
	commandList->SetGraphicsRootDescriptorTable(0, srvHeap->GetGPUDescriptorHandleForHeapStart());

	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	CD3DX12_RESOURCE_BARRIER resBarrierPresetToRT = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_PRESENT,
		D3D12_RESOURCE_STATE_RENDER_TARGET
	);
	commandList->ResourceBarrier(1, &resBarrierPresetToRT);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvHeapHandleIncrementSize);
	
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	commandList->DrawInstanced(3, 1, 0, 0);

	renderUI(commandList, rtvHandle);

	CD3DX12_RESOURCE_BARRIER resBarrierRTtoPresent = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	commandList->ResourceBarrier(1, &resBarrierRTtoPresent);

	return commandList;
}

bool CudaRasterizer::updateRenderTargetViews() {
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

	for (UINT i = 0; i < frameCount; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);
		device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvHandle);

		// Register the back buffer's resources manually since the resource manager doesn't own them, the swap chain does.
		backBuffersHandles[i] = resManager->registerResource(backBuffers[i].Get(), 1, D3D12_RESOURCE_STATE_PRESENT);

		rtvHandle.Offset(rtvHeapHandleIncrementSize);
	}

	return true;
}

void CudaRasterizer::timeIt() {
	using std::chrono::duration_cast;
	using HRC = std::chrono::high_resolution_clock;
	
	static constexpr double SECONDS_IN_NANOSECOND = 1e-9;
	static UINT64 frameCount = 0;
	static double elapsedTime = 0.0;
	static HRC clock;
	static HRC::time_point t0 = clock.now();

	HRC::time_point t1 = clock.now();
	deltaTime = (t1 - t0).count() * SECONDS_IN_NANOSECOND;
	elapsedTime += deltaTime;
	totalTime += deltaTime;
	
	++frameCount;
	t0 = t1;

	if (elapsedTime > 1.0) {
		fps = frameCount / elapsedTime;

		printf("FPS: %.2f\n", fps);

#if defined(D3D12_DEBUG)
		char buffer[512];
		sprintf_s(buffer, "FPS: %.2f\n", fps);
		OutputDebugString(buffer);
#endif // defined(D3D12_DEBUG)

		frameCount = 0;
		elapsedTime = 0.0;
	}
}

void CudaRasterizer::deinitCUDAData() {
	resManager->deregisterResource(dx12RenderTargetHandle);
	cudaRenderTargetDevice.deinitialize();

	vertexBuffer.deinitialize();
	indexBuffer.deinitialize();

	depthBuffer.deinitialize();

	for (int j = 0; j < MAX_RESOURCES_COUNT; ++j) {
		uavBuffers[j].deinitialize();
	}

	if (cudaRenderTargetHost != nullptr) {
		delete[] cudaRenderTargetHost;
		cudaRenderTargetHost = nullptr;
	}
}

void CudaRasterizer::initCUDAData() {
	deinitCUDAData();
	
	const CUDADevice &device = cudaManager->getDevices()[0];
	device.use();

	ResourceInitData resData(ResourceType::TextureBuffer);
	resData.textureData.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	resData.textureData.width = width;
	resData.textureData.height = height;
	resData.textureData.mipLevels = 1;
	resData.heapFlags = D3D12_HEAP_FLAG_NONE;
	dx12RenderTargetHandle = resManager->createBuffer(resData);
	cudaRenderTargetHost = new float[SizeType(width) * height * numComps];
	
	cudaRenderTargetDevice.initialize(SizeType(width) * height * numComps * sizeof(float));
	clearRenderTarget();
	CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	device.uploadConstantParam(&rt, "renderTarget");

	device.uploadConstantParam(&width, "width");
	device.uploadConstantParam(&height, "height");
}
