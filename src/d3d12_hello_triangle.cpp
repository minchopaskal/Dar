#include "d3d12_hello_triangle.h"

#include <cstdio>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include "asset_manager.h"
#include "d3dx12.h"
#include "defines.h"
#include "geometry.h"
#include "win32_app.h"

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height)},
	scissorRect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)},
	frameIndex(0),
	rtvHeapHandleIncrementSize(0),
	aspectRatio(width / real(height)),
	fenceEvent(NULL),
	fenceValue(0),
	vertexBufferView() { }

bool D3D12HelloTriangle::init() {
	if (!loadPipeline()) {
		return false;
	}
	
	return loadAssets();
}

void D3D12HelloTriangle::deinit() {
	gpuSync();
}

void D3D12HelloTriangle::update() {

}

void D3D12HelloTriangle::render() {
	populateCommandList();

	ID3D12CommandList *const commandLists[] = { commandList.Get() };
	commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	RETURN_ON_ERROR(swapChain->Present(0, 0), ,"Failed to execute command list!\n");

	gpuSync();
}

_Use_decl_annotations_
void GetHardwareAdapter(
		IDXGIFactory1 *pFactory,
		IDXGIAdapter1 **ppAdapter,
		bool requestHighPerformanceAdapter) {
	*ppAdapter = nullptr;

	ComPtr<IDXGIAdapter1> adapter;

	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
		for (
				UINT adapterIndex = 0;
				DXGI_ERROR_NOT_FOUND != factory6->EnumAdapterByGpuPreference(
			adapterIndex,
			requestHighPerformanceAdapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
			IID_PPV_ARGS(&adapter));
				++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
					// Don't select the Basic Render Driver adapter.
					// If you want a software adapter, pass in "/warp" on the command line.
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	} else {
		for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter); ++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
					// Don't select the Basic Render Driver adapter.
					// If you want a software adapter, pass in "/warp" on the command line.
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr))) {
				break;
			}
		}
	}

	*ppAdapter = adapter.Detach();
}

bool D3D12HelloTriangle::loadPipeline() {
#if defined(D3D12_DEBUG)
	/* Enable debug layer */
	{
		ComPtr<ID3D12Debug> debugLayer;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
			debugLayer->EnableDebugLayer();
			debugLayer->Release();
		}
	}
#endif // defined(D3D12_DEBUG)

	/* Create the device */
	ComPtr<IDXGIFactory4> dxgiFactory;
	RETURN_FALSE_ON_ERROR(
		CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)),
		"Error while creating DXGI Factory!\n"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	GetHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter, false);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
		"Failed to create device!\n"
	);

	/* Create command queue */
	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	cqDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

	RETURN_FALSE_ON_ERROR(
	device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&commandQueue)),
		"Failed to create command queue!\n"
	);

	/* Create a swap chain */
	DXGI_SWAP_CHAIN_DESC scDesc = {};
	scDesc.BufferCount = frameCount;
	scDesc.BufferDesc.Width = width;
	scDesc.BufferDesc.Height = height;
	scDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scDesc.OutputWindow = Win32App::getWindow();
	scDesc.SampleDesc.Count = 1;
	scDesc.Windowed = TRUE;

	//DXGI_SWAP_CHAIN_FULLSCREEN_DESC fscDesc = {};
	//fscDesc.RefreshRate = { 0, 0 };
	//fscDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE;
	////fscDesc.Scaling = DXGI_MODE_SCALING_STRETCHED;

	ComPtr<IDXGISwapChain> swapChainPlaceholder;
		
	RETURN_FALSE_ON_ERROR(
		dxgiFactory->CreateSwapChain(
			commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
			&scDesc,
			&swapChainPlaceholder
		),
		"Failed to create swap chain!\n"
	);
	
	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!\n"
	);

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	/* Create a descriptor heap */
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { };
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.NumDescriptors = frameCount;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	rtvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)),
		"Failed to create descriptor heap!\n"
	);
	rtvHeapHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

	/* Create render target view */
	{
		CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart());

		for (UINT i = 0; i < frameCount; ++i) {
			RETURN_FALSE_ON_ERROR_FMT(
				swapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i])),
				"Failed to create Render-Target-View for buffer %u!\n", i
			);
			device->CreateRenderTargetView(renderTargets[i].Get(), nullptr, rtvHandle);
			rtvHandle.Offset(1, rtvHeapHandleIncrementSize);
		}
	}

	/* Lastly, create the command allocator */
	RETURN_FALSE_ON_ERROR(
		device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)),
		"Failed to create command allocator!\n"
	);

	return true;
}

bool D3D12HelloTriangle::loadAssets() {
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

	/* Load & Compile the shaders */
	{
		ComPtr<ID3DBlob> vertexShader;
		ComPtr<ID3DBlob> pixelShader;

#if defined(D3D12_DEBUG)
		UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
		UINT compileFlags = 0;
#endif // defined(D3D12_DEBUG)

		RETURN_FALSE_ON_ERROR(
			D3DCompileFromFile(
				getAssetFullPath(L"basic.hlsl", AssetType::shader).c_str(),
				nullptr,
				nullptr,
				"VSMain",
				"vs_5_0",
				compileFlags,
				0,
				&vertexShader,
				nullptr
			),
			"Failed to compile vertex shader!\n"
		);
		RETURN_FALSE_ON_ERROR(
			D3DCompileFromFile(
				getAssetFullPath(L"basic.hlsl", AssetType::shader).c_str(),
				nullptr,
				nullptr,
				"PSMain",
				"ps_5_0",
				compileFlags,
				0,
				&pixelShader,
				nullptr
			),
			"Failed to compile pixel shader!\n"
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

	/* Create the command list */
	RETURN_FALSE_ON_ERROR(
		device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), pipelineState.Get(), IID_PPV_ARGS(&commandList)),
		"Failed to create command list!\n"
	);

	RETURN_FALSE_ON_ERROR(
		commandList->Close(),
		"Failed to close the command list after creation!\n"
	);

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

	/* Create fences and upload data to the GPU */
	{
		RETURN_FALSE_ON_ERROR(
			device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
			"Failed to create the fence!\n"
		);
		fenceValue = 1;

		fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
		if (fenceEvent == nullptr) {
			DWORD lastError = GetLastError();
			RETURN_FALSE_ON_ERROR_FMT(HRESULT_FROM_WIN32(lastError), "Last error error code: %d\n", lastError);
		}
	}

	gpuSync();

	return true;
}

void D3D12HelloTriangle::populateCommandList() {
	RETURN_ON_ERROR(commandAllocator->Reset(), , "Failed to reset the command allocator!\n");

	RETURN_ON_ERROR(commandList->Reset(commandAllocator.Get(), pipelineState.Get()), , "Failed to reset the command list!\n");

	commandList->SetPipelineState(pipelineState.Get());
	commandList->SetGraphicsRootSignature(rootSignature.Get());
	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &scissorRect);

	CD3DX12_RESOURCE_BARRIER resBarrierPresetToRT = CD3DX12_RESOURCE_BARRIER::Transition(
		renderTargets[frameIndex].Get(),
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
		renderTargets[frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	commandList->ResourceBarrier(1, &resBarrierRTtoPresent);

	commandList->Close();
}

void D3D12HelloTriangle::gpuSync() {
	UINT64 fenceVal = fenceValue;
	RETURN_ON_ERROR(commandQueue->Signal(fence.Get(), fenceVal), , "Failed to signal command queue!\n");
	fenceValue++;
	
	while (fence->GetCompletedValue() < fenceVal) {
		RETURN_ON_ERROR(
			fence->SetEventOnCompletion(fenceVal, fenceEvent), ,
			"Failed to set fence event on completion!\n"
		);
		WaitForSingleObject(fenceEvent, INFINITE);
	}

	frameIndex = swapChain->GetCurrentBackBufferIndex();
}
