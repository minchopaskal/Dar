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

// TODO: To make things simple, child projects should not rely on third party software.
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

#include "imgui.h"

CudaRasterizer::CudaRasterizer(Vector<String> &shaders, const String &windowTitle, UINT width, UINT height) :
	D3D12App(width, height, windowTitle.c_str()),
	rtvHeapHandleIncrementSize(0),
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, LONG_MAX, LONG_MAX }, // always render on the entire screen
	aspectRatio(width / static_cast<float>(height)),
	cudaRenderTargetHost{ nullptr },
	dx12RenderTargetHandle(INVALID_RESOURCE_HANDLE),
	fenceValues{ 0 },
	previousFrameIndex(0),
	fps(0.0),
	totalTime(0.0),
	deltaTime(0.0)
{
	inited = false;
	shaders.insert(shaders.begin(), "data\\rasterizer.ptx");
	shaders.insert(shaders.begin(), "data\\rasterizer_utils.ptx");
	if (!initializeCUDAManager(shaders, true)) {
		return;
	}
	cudaDevice = &getCUDAManager().getDevices()[0];

	// Switch to the context of this device from now on.
	if (cudaDevice->use().hasError()) {
		return;
	}

	init();

	inited = true;
}

CudaRasterizer::~CudaRasterizer() {
	deinit();
}

void CudaRasterizer::setUpdateFramebufferCallback(const UpdateFrameCallback cb, void *state) {
	updateFrameCb = cb;
	frameState = state;
}

void CudaRasterizer::setImGuiCallback(const DrawUICallback cb) {
	drawUICb = cb;
}

bool CudaRasterizer::isInitialized() const {
	return inited;
}

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

void CudaRasterizer::deinit() {
	flush();
	deinitCUDAData();
	deinitializeCUDAManager();
	Super::deinit();
}

void CudaRasterizer::update() {
	timeIt();

	if (updateFrameCb) {
		updateFrameCb(*this, frameState);
	}

	const CUstream stream = cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution);
	cuStreamSynchronize(stream);

	cudaRenderTargetDevice.download(cudaRenderTargetHost);
}

void CudaRasterizer::render() {
	// Upload the rendered image as a D3D12 texture
	const UploadHandle handle = resManager->beginNewUpload();
	D3D12_SUBRESOURCE_DATA subresData;
	subresData.pData = cudaRenderTargetHost;
	subresData.RowPitch = static_cast<LONG_PTR>(width) * numComps * sizeof(float);
	subresData.SlicePitch = subresData.RowPitch * height;
	resManager->uploadTextureData(handle, dx12RenderTargetHandle, &subresData, 1, 0);
	resManager->uploadBuffers();

	// Render a screen-quad with the texture
	CommandList cmdList = populateCommandList();
	commandQueueDirect.addCommandListForExecution(std::move(cmdList));

	fenceValues[frameIndex] = commandQueueDirect.executeCommandLists();

	const UINT syncInterval = vSyncEnabled ? 1 : 0;
	const UINT presentFlags = allowTearing && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
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
	aspectRatio = width / static_cast<float>(height);

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
	
}

void CudaRasterizer::onMouseScroll(double xOffset, double yOffset) {
	
}

void CudaRasterizer::drawUI() {
	Super::drawUI();

	if (drawUICb) {
		drawUICb();
	}
}

CUDAError CudaRasterizer::setUseDepthBuffer(bool useDepthBuffer) {
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam<bool>(&useDepthBuffer, "useDepthBuffer"));

	CUDAMemHandle depthBufferHandle = NULL;
	if (useDepthBuffer) {
		if (depthBuffer.getSize() == static_cast<SizeType>(width) * height * sizeof(float)) {
			depthBufferHandle = depthBuffer.handle();
		} else {
			depthBuffer.initialize(static_cast<SizeType>(width) * height * sizeof(float));
			depthBufferHandle = depthBuffer.handle();
			clearDepthBuffer();
		}
	}

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&depthBufferHandle, "depthBuffer"));

	return CUDAError();
}

CUDAError CudaRasterizer::setVertexBuffer(const Vertex* buffer, SizeType verticesCount) {
	vertexBuffer.initialize(verticesCount * sizeof(Vertex));
	vertexBuffer.upload(buffer);

	auto *vs = new Vertex[verticesCount];
	cuMemcpyDtoH(reinterpret_cast<void*>(vs), vertexBuffer.handle(), vertexBuffer.getSize());

	const CUDAMemHandle vertexBufferHandle = vertexBuffer.handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&vertexBufferHandle, "vertexBuffer"));
	
	cacheVerticesCount = verticesCount;

	return CUDAError();
}

CUDAError CudaRasterizer::setIndexBuffer(const unsigned int* buffer, SizeType indicesCount) {
	indexBuffer.initialize(indicesCount * sizeof(unsigned int));
	indexBuffer.upload(reinterpret_cast<const void*>(buffer));

	const CUDAMemHandle indexBufferHandle = indexBuffer.handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&indexBufferHandle, "indexBuffer"));
	return CUDAError();
}

CUDAError CudaRasterizer::setUavBuffer(const void *buffer, SizeType size, int index) {
	if (index < 0 || index >= MAX_RESOURCES_COUNT) {
		return CUDAError(CUDA_ERROR_INVALID_VALUE, "Invalid UAV index", "Index of UAV resource is out of bounds!");
	}

	uavBuffers[index].initialize(size);
	uavBuffers[index].upload(buffer);

	const CUDAMemHandle uavBufferHandle = uavBuffers[index].handle();

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&uavBufferHandle, "resources", static_cast<SizeType>(index)));
	return CUDAError();
}

CUDAError CudaRasterizer::setShaderProgram(const String &name) const {
	CUDADefaultBuffer shaderPtrs;
	RETURN_ON_CUDA_ERROR_HANDLED(shaderPtrs.initialize(sizeof(CUDAShaderPointers)));

	CUDAFunction getShaderPtrsFunction(cudaDevice->getModule(), ("getShaderPtrs_" + name).c_str());
	getShaderPtrsFunction.addParams(
		shaderPtrs.handle()
	);
	RETURN_ON_CUDA_ERROR_HANDLED(getShaderPtrsFunction.launchSync(1, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	CUDAShaderPointers ptrs = { };
	RETURN_ON_CUDA_ERROR_HANDLED(shaderPtrs.download(&ptrs));
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&ptrs.vsShaderPtr, "vsShader"));
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&ptrs.psShaderPtr, "psShader"));

	return CUDAError();
}

CUDAError CudaRasterizer::drawIndexed(const unsigned int numPrimitives) const {
	constexpr unsigned int verticesInTriangle = 3;

	CUDAFunction vertexProcessingFunc(cudaDevice->getModule(), "processVertices");
	vertexProcessingFunc.addParams(
		cacheVerticesCount,
		width,
		height
	);
	RETURN_ON_CUDA_ERROR_HANDLED(vertexProcessingFunc.launchSync(numPrimitives * verticesInTriangle, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	CUDAFunction drawIndexedFunc(cudaDevice->getModule(), "drawIndexed");
	drawIndexedFunc.addParams(
		numPrimitives,
		width,
		height
	);
	
	RETURN_ON_CUDA_ERROR_HANDLED(drawIndexedFunc.launchSync(numPrimitives, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	return CUDAError();
}

CUDAError CudaRasterizer::setClearColor(const Vec4 &color) const {
	float colorHost[4];
	for (int i = 0; i < numComps; ++i) {
		colorHost[i] = color.data[i];
	}

	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantArray(colorHost, 4, "clearColor"));

	return CUDAError();
}

CUDAError CudaRasterizer::setCulling(CudaRasterizerCullType cullType) const {
	RETURN_ON_CUDA_ERROR_HANDLED(cudaDevice->uploadConstantParam(&cullType, "cullType"));
	return CUDAError();
}

CUDAError CudaRasterizer::clearRenderTarget() {
	const CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	RETURN_ON_CUDA_ERROR(cuMemsetD8(rt, 0, cudaRenderTargetDevice.getSize()));

	CUDAFunction blankKernel(cudaDevice->getModule(), "blank");
	blankKernel.addParams(
		rt,
		width,
		height
	);
	
	RETURN_ON_CUDA_ERROR_HANDLED(blankKernel.launchSync(width * height, cudaDevice->getDefaultStream(CUDADefaultStreamsEnumeration::Execution)));

	return CUDAError();
}

CUDAError CudaRasterizer::clearDepthBuffer() {
	if (depthBuffer.getSize() == 0) {
		return CUDAError();
	}

	constexpr float floatOne = 1.f;
	const auto floatOneBits = *reinterpret_cast<const unsigned int*>(&floatOne);

	const CUDAMemHandle depthBufferHandle = depthBuffer.handle();
	RETURN_ON_CUDA_ERROR(cuMemsetD32(depthBufferHandle, floatOneBits, static_cast<SizeType>(width) * height));

	return CUDAError();
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

	for (auto &uavBuffer : uavBuffers) {
		uavBuffer.deinitialize();
	}

	if (cudaRenderTargetHost != nullptr) {
		delete[] cudaRenderTargetHost;
		cudaRenderTargetHost = nullptr;
	}
}

void CudaRasterizer::initCUDAData() {
	deinitCUDAData();

	ResourceInitData resData(ResourceType::TextureBuffer);
	resData.textureData.format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	resData.textureData.width = width;
	resData.textureData.height = height;
	resData.textureData.mipLevels = 1;
	resData.heapFlags = D3D12_HEAP_FLAG_NONE;
	dx12RenderTargetHandle = resManager->createBuffer(resData);
	cudaRenderTargetHost = new float[static_cast<SizeType>(width) * height * numComps];
	
	cudaRenderTargetDevice.initialize(static_cast<SizeType>(width) * height * numComps * sizeof(float));
	clearRenderTarget();
	CUDAMemHandle rt = cudaRenderTargetDevice.handle();
	cudaDevice->uploadConstantParam(&rt, "renderTarget");

	cudaDevice->uploadConstantParam(&width, "width");
	cudaDevice->uploadConstantParam(&height, "height");
}
