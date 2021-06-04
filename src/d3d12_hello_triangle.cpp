#include "d3d12_hello_triangle.h"

#include <algorithm>
#include <cstdio>
#include <chrono>

#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "asset_manager.h"
#include "defines.h"
#include "geometry.h"
#include "d3d12_app.h"
#include "d3d12_utils.h"

#include <glfw/glfw3.h> // keyboard input

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	rtvHeapHandleIncrementSize(0),
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, LONG_MAX, LONG_MAX }, // always render on the entire screen
	aspectRatio(width / real(height)),
	FOV(45.0),
	fps(0.0),
	totalTime(0.0)
{ }

int D3D12HelloTriangle::init() {
	if (!D3D12App::init()) {
		return false;
	}

	/* Create a descriptor heap for RTVs */
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { };
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.NumDescriptors = frameCount;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	// TODO: 0 since using only one device. Look into that.
	rtvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)),
		"Failed to create RTV descriptor heap!\n"
	);

	rtvHeapHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	
	D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = { };
	dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
	dsvHeapDesc.NumDescriptors = 1;
	dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	// TODO: 0 since using only one device. Look into that.
	dsvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap)),
		"Failed to create DSV descriptor heap!\n"
	);
	
	if (!resizeDepthBuffer(this->width, this->height)) {
		return false;
	}

	if (!updateRenderTargetViews()) {
		return false;
	}

	return true;
}

void D3D12HelloTriangle::deinit() {
	flush();
}

void D3D12HelloTriangle::update() {
	timeIt();

	// Update MVP matrices
	float angle = static_cast<float>(totalTime * 90.0);
	const Vec3 rotationAxis = Vec3(0, 1, 1);
	modelMat = Mat4(1.f);
	modelMat = glm::rotate(modelMat, glm::radians(angle), rotationAxis);

	const Vec3 eyePosition = Vec3(0, 0, -10);
	const Vec3 focusPoint  = Vec3(0, 0, 0);
	const Vec3 upDirection = Vec3(0, 1, 0);
	viewMat = glm::lookAtLH(eyePosition, focusPoint, upDirection);

	projectionMat = glm::perspectiveFovLH(FOV, float(this->width), float(this->height), 0.1f, 100.f);
}

void D3D12HelloTriangle::render() {
	ComPtr<ID3D12GraphicsCommandList2> cmdList = populateCommandList();
	fenceValues[frameIndex] = commandQueueDirect.executeCommandList(cmdList);

	UINT syncInterval = vSyncEnabled ? 1 : 0;
	UINT presentFlags = tearingEnabled && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), ,"Failed to execute command list!\n");
	
	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// wait for the next frame's buffer
	commandQueueDirect.waitForFenceValue(fenceValues[frameIndex]);
}

void D3D12HelloTriangle::onResize(int w, int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1, w);
	this->height = std::max(1, h);
	viewport = { 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) };
	aspectRatio = width / real(height);

	flush();

	for (int i = 0; i < frameCount; ++i) {
		backBuffers[i].Reset();
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

	resizeDepthBuffer(this->width, this->height);
}

void D3D12HelloTriangle::onKeyboardInput(int key, int action) {
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	vSyncEnabled = keyPressed[GLFW_KEY_V];
}

void D3D12HelloTriangle::onMouseScroll(double xOffset, double yOffset) {
	static const float speed = 10.f;
	FOV -= speed * deltaTime * yOffset;
	FOV = std::min(std::max(30.f, FOV), 120.f);
}

int D3D12HelloTriangle::loadAssets() {
	/* Create root signature */
	{
		D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};
		featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
		if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
			featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
		}

		// Allow input layout and deny unnecessary access to certain pipeline stages.
		D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
			D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;

		CD3DX12_ROOT_PARAMETER1 rootParameters[1] = { {} };
		rootParameters[0].InitAsConstants(sizeof(Mat4) / sizeof(float), 0, 0, D3D12_SHADER_VISIBILITY_VERTEX);

		CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
		rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		// TODO: read error if any
		RETURN_FALSE_ON_ERROR(
			D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, &signature, &error),
			"Failed to create root signature!\n"
		);

		RETURN_FALSE_ON_ERROR(
			device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature)),
			"Failed to create root signature!\n"
		);
	}

	/* Load the shaders */
	{
		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;

#if defined(D3D12_DEBUG)
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
		UINT compileFlags = 0;
#endif // defined(D3D12_DEBUG)

		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath(L"basic_vs.cso", AssetType::shader).c_str(),
				&vertexShader
			),
			"Failed to load vertex shader!\n"
		);

		RETURN_FALSE_ON_ERROR(
			D3DReadFileToBlob(
				getAssetFullPath(L"basic_ps.cso", AssetType::shader).c_str(),
				&pixelShader
			),
			"Failed to load pixel shader!\n"
		);

		// TODO: find a better way 
		// (idea: struct with all tokens - set all wanted tokens and init a stream object with all non null tokens)
		PipelineStateStream pipelineStateStream;
		
		CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE rootSignatureToken = rootSignature.Get();
		pipelineStateStream.insert(rootSignatureToken);

		D3D12_INPUT_ELEMENT_DESC inputLayouts[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		};
		CD3DX12_PIPELINE_STATE_STREAM_INPUT_LAYOUT inputLayoutToken;
		inputLayoutToken = { inputLayouts, _countof(inputLayouts) };
		pipelineStateStream.insert(inputLayoutToken);

		CD3DX12_PIPELINE_STATE_STREAM_VS vsToken= CD3DX12_SHADER_BYTECODE(vertexShader.Get());
		pipelineStateStream.insert(vsToken);

		CD3DX12_PIPELINE_STATE_STREAM_PS psToken = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
		pipelineStateStream.insert(psToken);

		CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY topologyToken = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		pipelineStateStream.insert(topologyToken);

		D3D12_RT_FORMAT_ARRAY rtFormat = {};
		rtFormat.NumRenderTargets = 1;
		rtFormat.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS rtFormatToken = rtFormat;
		pipelineStateStream.insert(rtFormatToken);

		CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT dsFormatToken = DXGI_FORMAT_D32_FLOAT;
		pipelineStateStream.insert(dsFormatToken);
		
		D3D12_PIPELINE_STATE_STREAM_DESC pipelineDesc = {};
		pipelineDesc.pPipelineStateSubobjectStream = pipelineStateStream.get();
		pipelineDesc.SizeInBytes = pipelineStateStream.size();

		RETURN_FALSE_ON_ERROR(
			device->CreatePipelineState(&pipelineDesc, IID_PPV_ARGS(&pipelineState)),
			"Failed to create pipeline state!\n"
		);
	}

	/* Create and copy data to the vertex buffer*/
	{
		static Vertex triangleVertices[] = {
			{ {  0.0f,  1.5f, 0.0f, 1.f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
			{ {  1.5f, -1.5f, 0.0f, 1.f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
			{ { -1.5f, -1.5f, 0.0f, 1.f }, { 0.0f, 0.0f, 1.0f, 1.0f } }
		};
		const UINT vertexBufferSize = sizeof(triangleVertices);
		CPUBuffer cpuTriangleVertexBuffer = {
			triangleVertices,
			vertexBufferSize
		};

		static WORD triangleIndices[] = { 0, 1, 2 };
		const UINT indexBufferSize = sizeof(triangleIndices);
		CPUBuffer cpuTriangleIndexBuffer = {
			triangleIndices,
			indexBufferSize
		};

		ComPtr<ID3D12GraphicsCommandList2> commandList = commandQueueCopy.getCommandList();

		ComPtr<ID3D12Resource> stagingVertexBuffer;
		if (!updateBufferResource(
			device,
			commandList,
			&vertexBuffer,
			&stagingVertexBuffer,
			cpuTriangleVertexBuffer,
			D3D12_RESOURCE_FLAG_NONE,
			D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
		)) {
			return false;
		}

		ComPtr<ID3D12Resource> stagingIndexBuffer;
		if (!updateBufferResource(
			device,
			commandList,
			&indexBuffer,
			&stagingIndexBuffer,
			cpuTriangleIndexBuffer,
			D3D12_RESOURCE_FLAG_NONE,
			D3D12_RESOURCE_STATE_INDEX_BUFFER
			)) {
			return false;
		}

		int fenceVal = commandQueueCopy.executeCommandList(commandList);
		commandQueueCopy.waitForFenceValue(fenceVal);

		vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
		vertexBufferView.SizeInBytes = vertexBufferSize;
		vertexBufferView.StrideInBytes = sizeof(Vertex);

		indexBufferView.BufferLocation = indexBuffer->GetGPUVirtualAddress();
		indexBufferView.SizeInBytes = indexBufferSize;
		indexBufferView.Format = DXGI_FORMAT_R16_UINT;
	}

	return true;
}

ComPtr<ID3D12GraphicsCommandList2> D3D12HelloTriangle::populateCommandList() {
	ComPtr<ID3D12GraphicsCommandList2> commandList = commandQueueDirect.getCommandList();

	commandList->SetPipelineState(pipelineState.Get());
	commandList->SetGraphicsRootSignature(rootSignature.Get());
	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	CD3DX12_RESOURCE_BARRIER resBarrierPresetToRT = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_PRESENT,
		D3D12_RESOURCE_STATE_RENDER_TARGET
	);
	commandList->ResourceBarrier(1, &resBarrierPresetToRT);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvHeapHandleIncrementSize);
	D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();

	const float blue[] = { 0.2f, 0.2f, 0.8f, 1.f };
	const float red[] = { 1.f, 0.2f, 0.2f, 1.f };
	const float *clearColor = blue;
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);
	
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
	commandList->IASetIndexBuffer(&indexBufferView);

	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
	Mat4 mvp = projectionMat * viewMat * modelMat;
	commandList->SetGraphicsRoot32BitConstants(0, sizeof(Mat4) / sizeof(float), &mvp[0], 0);
	commandList->DrawIndexedInstanced(3, 1, 0, 0, 0);

	CD3DX12_RESOURCE_BARRIER resBarrierRTtoPresent = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	commandList->ResourceBarrier(1, &resBarrierRTtoPresent);

	return commandList;
}

bool D3D12HelloTriangle::updateRenderTargetViews() {
	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

	for (UINT i = 0; i < frameCount; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!\n", i
		);
		device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvHandle);
		rtvHandle.Offset(rtvHeapHandleIncrementSize);
	}

	return true;
}

bool D3D12HelloTriangle::resizeDepthBuffer(int width, int height) {
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
		"Failed to create/resize depth buffer!\n"
	);

	D3D12_DEPTH_STENCIL_VIEW_DESC dsDesc = {};
	dsDesc.Format = DXGI_FORMAT_D32_FLOAT;
	dsDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	dsDesc.Flags = D3D12_DSV_FLAG_NONE;
	dsDesc.Texture2D.MipSlice = 0;

	device->CreateDepthStencilView(depthBuffer.Get(), &dsDesc, dsvHeap->GetCPUDescriptorHandleForHeapStart());

	return true;
}

void D3D12HelloTriangle::timeIt() {
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
