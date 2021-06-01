#include "d3d12_hello_triangle.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "asset_manager.h"
#include "d3dx12.h"
#include "defines.h"
#include "geometry.h"
#include "d3d12_app.h"

#include <glfw/glfw3.h> // keyboard input

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	vertexBufferView{ },
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) },
	aspectRatio(width / real(height)),
	fps(0.0)
{ }

int D3D12HelloTriangle::init() {
	return D3D12App::init();
}

void D3D12HelloTriangle::deinit() {
	commandQueueDirect.flush();
}

void D3D12HelloTriangle::update() {
	timeIt();
}

void D3D12HelloTriangle::render() {
	ComPtr<ID3D12GraphicsCommandList2> cmdList = populateCommandList();
	fenceValues[frameIndex] = commandQueueDirect.executeCommandList(cmdList);

	UINT syncInterval = vSyncEnabled ? 1 : 0;
	UINT presentFlags = tearingEnabled && !vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), ,"Failed to execute command list!\n");
	
	frameIndex = swapChain->GetCurrentBackBufferIndex();
	commandQueueDirect.waitForFenceValue(fenceValues[frameIndex]);
}

void D3D12HelloTriangle::resize(int w, int h) {
	if (width == w && height == h) {
		return;
	}

	this->width = std::max(1, w);
	this->height = std::max(1, h);
	viewport = { 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) };
	scissorRect = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
	aspectRatio = width / real(height);

	commandQueueDirect.flush();

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
}

void D3D12HelloTriangle::keyboardInput(int key, int action) {
	keyPressed[key] = !(action == GLFW_RELEASE);
	keyRepeated[key] = (action == GLFW_REPEAT);

	// TODO: mapping keys to engine actions
	if (keyPressed[GLFW_KEY_F] && !keyRepeated[GLFW_KEY_F]) {
		toggleFullscreen();
	}

	vSyncEnabled = keyPressed[GLFW_KEY_V];
}

int D3D12HelloTriangle::loadAssets() {
	/* Create empty root signature */
	{
		CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc = {};
		rootSignatureDesc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		// TODO: read error if any
		RETURN_FALSE_ON_ERROR(
			D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error),
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

		D3D12_INPUT_ELEMENT_DESC inputElemDesc[] = {
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		};

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psDesc = { };
		psDesc.pRootSignature = rootSignature.Get();
		psDesc.VS = { reinterpret_cast<UINT8*>(vertexShader->GetBufferPointer()), vertexShader->GetBufferSize() };
		psDesc.PS = { reinterpret_cast<UINT8*>(pixelShader->GetBufferPointer()), pixelShader->GetBufferSize() };
		psDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psDesc.SampleMask = UINT_MAX;
		psDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psDesc.DepthStencilState.DepthEnable = FALSE;
		psDesc.DepthStencilState.StencilEnable = FALSE;
		psDesc.InputLayout = { inputElemDesc, _countof(inputElemDesc) };
		psDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psDesc.NumRenderTargets = 1;
		psDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		psDesc.SampleDesc.Count = 1;
		RETURN_FALSE_ON_ERROR(
			device->CreateGraphicsPipelineState(&psDesc, IID_PPV_ARGS(&pipelineState)),
			"Failed to create pipeline state!\n"
		);
	}

	/* Create and copy data to the vertex buffer*/
	{
		Vertex triangleVertices[] =
		{
				{ { 0.f, .5f, 0.0f, 1.f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
				{ { .5f, -.5f, 0.0f, 1.f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
				{ { -.5f, -.5f, 0.0f, 1.f }, { 0.0f, 0.0f, 1.0f, 1.0f } },
		};

		const UINT vertexBufferSize = sizeof(triangleVertices);

		CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
		CD3DX12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
		RETURN_FALSE_ON_ERROR(
			device->CreateCommittedResource(
				&heapProps,
				D3D12_HEAP_FLAG_NONE,
				&resDesc,
				D3D12_RESOURCE_STATE_GENERIC_READ,
				nullptr,
				IID_PPV_ARGS(&vertexBuffer)
			),
			"Failed to create vertex buffer!\n"
		);

		// copy vertex buffer data
		UINT8 *vertexBufferData;
		D3D12_RANGE readRange{ 0, 0 };
		RETURN_FALSE_ON_ERROR(
			vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&vertexBufferData)),
			"Failed to map vertex buffer to host!\n"
		);
		memcpy(vertexBufferData, triangleVertices, vertexBufferSize);
		vertexBuffer->Unmap(0, nullptr);

		vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
		vertexBufferView.SizeInBytes = vertexBufferSize;
		vertexBufferView.StrideInBytes = sizeof(Vertex);
	}

	commandQueueDirect.flush();

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
	commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	const float clearColor[] = { 0.2f, 0.2f, 0.8f, 1.f };
	commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	
	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
	commandList->DrawInstanced(3, 1, 0, 0);

	CD3DX12_RESOURCE_BARRIER resBarrierRTtoPresent = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	commandList->ResourceBarrier(1, &resBarrierRTtoPresent);

	return commandList;
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
	std::chrono::nanoseconds deltaTime = t1 - t0;
	elapsedTime += deltaTime.count() * SECONDS_IN_NANOSECOND;
	
	++frameCount;
	t0 = t1;

	if (elapsedTime > 1.0) {
		fps = frameCount / elapsedTime;

#if defined(D3D12_DEBUG)
		char buffer[512];
		sprintf_s(buffer, "FPS: %.2f\n", fps);
		OutputDebugString(buffer);
#endif // defined(D3D12_DEBUG)

		frameCount = 0;
		elapsedTime = 0.0;
	}
}
