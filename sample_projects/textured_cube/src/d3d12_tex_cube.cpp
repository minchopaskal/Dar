#include "d3d12_tex_cube.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "d3d12_app.h"
#include "d3d12_asset_manager.h"
#include "d3d12_pipeline_state.h"
#include "d3d12_res_tracker.h"
#include "d3d12_utils.h"
#include "geometry.h"

// TODO: To make things simple, child projects should not rely on third party software
// Expose some input controller interface or something like that.
#include <glfw/glfw3.h> // keyboard input

// For loading the texture image
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb_image.h"

struct ImageData {
	void *data = nullptr;
	int width = 0;
	int height = 0;
	int ncomp = 0;

	operator CPUBuffer() {
		return CPUBuffer{
			data,
			width * height * UINT64(ncomp)
		};
	}
};

ImageData loadImage(const wchar_t *name) {
	WString pathStr = getAssetFullPath(name, AssetType::Texture);
	const wchar_t *path = pathStr.c_str();
	const size_t bufferlen = wcslen(path) * sizeof(wchar_t);
	char *pathUTF8 = new char[bufferlen + 1];
	stbi_convert_wchar_to_utf8(pathUTF8, bufferlen, path);

	ImageData result = {};
	result.data = stbi_load(pathUTF8, &result.width, &result.height, nullptr, 4);
	result.ncomp = 4;

	return result;
}

D3D12TexturedCube::D3D12TexturedCube(UINT width, UINT height, const String &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	rtvHeapHandleIncrementSize(0),
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, LONG_MAX, LONG_MAX }, // always render on the entire screen
	aspectRatio(width / float(height)),
	fenceValues{ 0 },
	FOV(45.0),
	fps(0.0),
	totalTime(0.0)
{ }

int D3D12TexturedCube::init() {
	if (!D3D12App::init()) {
		return false;
	}

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
	
	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = { };
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	dsvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap)),
		"Failed to create DSV descriptor heap!"
	);
	
	if (!resizeDepthBuffer(this->width, this->height)) {
		return false;
	}

	if (!updateRenderTargetViews()) {
		return false;
	}

	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.NumDescriptors = numTextures;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	srvHeapDesc.NodeMask = 0;
	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap)),
		"Failed to create DSV descriptor heap!"
	);

	return true;
}

void D3D12TexturedCube::deinit() {
	flush();
}

void D3D12TexturedCube::update() {
	timeIt();

	// Update MVP matrices
	float angle = static_cast<float>(totalTime * 90.0);
	const Vec3 rotationAxis = Vec3(0, 1, 1);
	Mat4 modelMat = Mat4(1.f);
	modelMat = modelMat.rotate(rotationAxis, angle);
	modelMat = modelMat.translate({ 0, 0, 0 });

	const Vec3 eyePosition = Vec3(0, 0, -10);
	const Vec3 focusPoint  = Vec3(0, 0, 0);
	const Vec3 upDirection = Vec3(0, 1, 0);
	Mat4 viewMat = dmath::lookAt(focusPoint, eyePosition, upDirection);
	Mat4 projectionMat = projectionType == ProjectionType::Perspective ? 
		dmath::perspective(FOV, aspectRatio, 0.1f, 100.f) :
		dmath::orthographic(-orthoDim * aspectRatio, orthoDim * aspectRatio, -orthoDim, orthoDim, 0.1f, 100.f);
	MVP = projectionMat * viewMat * modelMat;

	// We know for sure that the MVP buffer for the current `frameIndex` is not in use
	// since we waited for the fence for the current `frameIndex` at the end of the previous call to render()
	//MVPcb[frameIndex].Reset();

	CommandList commandList = commandQueueCopy.getCommandList();
	ComPtr<ID3D12Resource> stagingBuffer;
	updateBufferResource(
		device,
		commandList.getComPtr(),
		&MVPcb[frameIndex],
		&stagingBuffer,
		{ &MVP, sizeof(Mat4) },
		D3D12_RESOURCE_FLAG_NONE
	);
	commandQueueCopy.addCommandListForExecution(std::move(commandList));
	auto fenceVal = commandQueueCopy.executeCommandLists();
	commandQueueCopy.waitForFenceValue(fenceVal);
}

void D3D12TexturedCube::render() {
	commandQueueDirect.addCommandListForExecution(populateCommandList());
	fenceValues[frameIndex] = commandQueueDirect.executeCommandLists();

	UINT syncInterval = vSyncEnabled ? 1 : 0;
	UINT presentFlags = allowTearing && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), ,"Failed to execute command list!");
	
	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// wait for the next frame's buffer
	commandQueueDirect.waitForFenceValue(fenceValues[frameIndex]);
}

void D3D12TexturedCube::onResize(int w, int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1, w);
	this->height = std::max(1, h);
	viewport = { 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) };
	aspectRatio = width / float(height);

	flush();

	for (int i = 0; i < frameCount; ++i) {
		ResourceTracker::unregisterResource(backBuffers[i].Get());
		backBuffers[i].Reset();
	}
	depthBuffer.Reset();

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

	resizeDepthBuffer(this->width, this->height);
}

void D3D12TexturedCube::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	if (keyPressed[GLFW_KEY_V] && !keyRepeated[GLFW_KEY_V]) {
		vSyncEnabled = !vSyncEnabled;
	}

	if (keyPressed[GLFW_KEY_O] && !keyRepeated[GLFW_KEY_O]) {
		projectionType = static_cast<ProjectionType>((static_cast<int>(projectionType) + 1) % 2);
	}
}

void D3D12TexturedCube::onMouseScroll(double xOffset, double yOffset) {
	static const double speed = 5.f;
	float change = float(yOffset);
	if (projectionType == ProjectionType::Perspective) {
		FOV -= change;
		FOV = dmath::min(dmath::max(30.f, FOV), 120.f);
	} else {
		orthoDim -= change;
		orthoDim = dmath::min(dmath::max(1.f, orthoDim), 100.f);
	}
}

int D3D12TexturedCube::loadAssets() {
	D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
	};

	CD3DX12_STATIC_SAMPLER_DESC sampler{ D3D12_FILTER_MIN_MAG_MIP_POINT };
	PipelineStateDesc psDesc = {};
	psDesc.shaderName = L"basic";
	psDesc.shadersMask = sif_useVertex;
	psDesc.inputLayouts = inputLayouts;
	psDesc.staticSamplerDesc = &sampler;
	psDesc.numInputLayouts = _countof(inputLayouts);
	psDesc.numConstantBufferViews = 1;
	psDesc.numTextures = numTextures;
	psDesc.maxVersion = rootSignatureFeatureData.HighestVersion;
	if (!pipelineState.init(device, psDesc)) {
		return false;
	}

	/* Create and copy data to the vertex buffer*/
	{
		static Vertex cubeVertices[] = {
			{ {-1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f } }, // 0
			{ {-1.0f,  1.0f, -1.0f }, { 1.0f, 1.0f } }, // 1
			{ { 1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f } }, // 2
			{ { 1.0f, -1.0f, -1.0f }, { 0.0f, 0.0f } }, // 3
			{ {-1.0f, -1.0f,  1.0f }, { 0.0f, 0.0f } }, // 4
			{ {-1.0f,  1.0f,  1.0f }, { 0.0f, 1.0f } }, // 5
			{ { 1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f } }, // 6
			{ { 1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f } }, // 7
			
			{ {-1.0f,  1.0f, -1.0f }, { 1.0f, 0.0f } }, // 8
			{ { 1.0f,  1.0f, -1.0f }, { 0.0f, 0.0f } }, // 9

			{ {-1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f } }, // 10
			{ { 1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f } }, // 11
		};
		const UINT vertexBufferSize = sizeof(cubeVertices);
		CPUBuffer cpuVertexBuffer = {
			cubeVertices,
			vertexBufferSize
		};

		static WORD cubeIndices[] = { 
			0, 1, 2, 0, 2, 3,
			4, 6, 5, 4, 7, 6,
			4, 5, 1, 4, 1, 0,
			3, 2, 6, 3, 6, 7,
			8, 5, 6, 1, 6, 9,
			4, 10, 11, 4, 11, 7
		};
		const UINT indexBufferSize = sizeof(cubeIndices);
		CPUBuffer cpuIndexBuffer = {
			cubeIndices,
			indexBufferSize
		};

		CommandList commandList = commandQueueCopy.getCommandList();

		ComPtr<ID3D12Resource> stagingVertexBuffer;
		if (!updateBufferResource(
			device,
			commandList.getComPtr(),
			&vertexBuffer,
			&stagingVertexBuffer,
			cpuVertexBuffer,
			D3D12_RESOURCE_FLAG_NONE
		)) {
			return false;
		}

		ComPtr<ID3D12Resource> stagingIndexBuffer;
		if (!updateBufferResource(
			device,
			commandList.getComPtr(),
			&indexBuffer,
			&stagingIndexBuffer,
			cpuIndexBuffer,
			D3D12_RESOURCE_FLAG_NONE
			)) {
			return false;
		}

		/* Load the texture */
		ImageData texData[numTextures];
		ComPtr<ID3D12Resource> stagingImageBuffers[numTextures];
		for (int i = 0; i < numTextures; ++i) {
			texData[i] = loadImage(L"box.jpg");
			if (!updateTex2DResource(
				device,
				commandList.getComPtr(),
				&textures[i],
				&stagingImageBuffers[i],
				texData[i],
				texData[i].width,
				texData[i].height,
				DXGI_FORMAT_R8G8B8A8_UINT,
				D3D12_RESOURCE_FLAG_NONE
			)) {
				return false;
			}
		}

		vertexBuffer->SetName(L"VertexBuffer");
		indexBuffer->SetName(L"IndexBuffer");
		textures[0]->SetName(L"Textures[0]");

		commandQueueCopy.addCommandListForExecution(std::move(commandList));
		UINT64 fenceVal = commandQueueCopy.executeCommandLists();
		commandQueueCopy.waitForFenceValue(fenceVal);

		vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
		vertexBufferView.SizeInBytes = vertexBufferSize;
		vertexBufferView.StrideInBytes = sizeof(Vertex);

		indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
		indexBufferView.SizeInBytes = indexBufferSize;
		indexBufferView.Format = DXGI_FORMAT_R16_UINT;

		SIZE_T srvHeapHandleSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		D3D12_CPU_DESCRIPTOR_HANDLE handle = srvHeap->GetCPUDescriptorHandleForHeapStart();
		for (int i = 0; i < numTextures; ++i) {
			D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
			srvDesc.Format = texData[i].ncomp == 4 ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_UNKNOWN;
			srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srvDesc.Texture2D.MipLevels = 1;

			device->CreateShaderResourceView(textures[i].Get(), &srvDesc, handle);
			handle.ptr += srvHeapHandleSize;
		}
	}

	return true;
}

CommandList D3D12TexturedCube::populateCommandList() {
	CommandList commandList = commandQueueDirect.getCommandList();

	if (!commandList.isValid()) {
		return commandList;
	}

	commandList->SetPipelineState(pipelineState.getPipelineState());

	static bool staticResStateChange = false;
	if (!staticResStateChange) {
		commandList.transition(textures[0], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	}
	commandList->SetDescriptorHeaps(1, srvHeap.GetAddressOf());

	commandList->SetGraphicsRootSignature(pipelineState.getRootSignature());

	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	commandList.transition(backBuffers[frameIndex], D3D12_RESOURCE_STATE_RENDER_TARGET);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvHeapHandleIncrementSize);
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();

	const float blue[] = { 0.2f, 0.2f, 0.8f, 1.f };
	const float red[] = { 1.f, 0.2f, 0.2f, 1.f };
	const float *clearColor = blue;
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);
	
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// TODO: this only needs to be done once for the vertex and index buffer so we need some sort of a resource tracking mechanism
	if (!staticResStateChange) {
		commandList.transition(vertexBuffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
		commandList.transition(indexBuffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
		staticResStateChange = true;
	}
	D3D12_RESOURCE_BARRIER copyToConst = CD3DX12_RESOURCE_BARRIER::Transition(
		MVPcb[frameIndex].Get(),
		D3D12_RESOURCE_STATE_COPY_DEST,
		D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
	);
	commandList->ResourceBarrier(1, &copyToConst);

	commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
	commandList->IASetIndexBuffer(&indexBufferView);
	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	commandList->SetGraphicsRootConstantBufferView(0, MVPcb[frameIndex]->GetGPUVirtualAddress());

	commandList->DrawIndexedInstanced(36, 1, 0, 0, 0);

	commandList.transition(backBuffers[frameIndex], D3D12_RESOURCE_STATE_PRESENT);

	return commandList;
}

bool D3D12TexturedCube::updateRenderTargetViews() {
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

	for (UINT i = 0; i < frameCount; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);
		device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(rtvHeapHandleIncrementSize);
		ResourceTracker::registerResource(backBuffers[i].Get(), Vector<D3D12_RESOURCE_STATES>(D3D12_RESOURCE_STATE_RENDER_TARGET));
	}

	return true;
}

bool D3D12TexturedCube::resizeDepthBuffer(int width, int height) {
	width = std::max(1, width);
	height = std::max(1, height);

	D3D12_CLEAR_VALUE clearValue = {};
	clearValue.Format = DXGI_FORMAT_D32_FLOAT;
	clearValue.DepthStencil = { 1.f, 0 };

	RETURN_FALSE_ON_ERROR(
		device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Tex2D(
				DXGI_FORMAT_D32_FLOAT, width, height,
				1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
			),
			D3D12_RESOURCE_STATE_DEPTH_WRITE,
			&clearValue,
			IID_PPV_ARGS(&depthBuffer)
		),
		"Failed to create/resize depth buffer!"
	);

	D3D12_DEPTH_STENCIL_VIEW_DESC dsDesc = {};
	dsDesc.Format = DXGI_FORMAT_D32_FLOAT;
	dsDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsDesc.Texture2D.MipSlice = 0;

	device->CreateDepthStencilView(depthBuffer.Get(), &dsDesc, dsvHeap->GetCPUDescriptorHandleForHeapStart());

	return true;
}

void D3D12TexturedCube::timeIt() {
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
