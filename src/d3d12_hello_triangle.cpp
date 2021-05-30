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
#include "win32_app.h"

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, const std::string &windowTitle) :
	D3D12App(width, height, windowTitle.c_str()),
	vertexBufferView{ },
	fenceEvent(NULL),
	fenceValue(0),
	frameFenceValues{ 0 },
	viewport{ 0.f, 0.f, static_cast<float>(width), static_cast<float>(height) },
	scissorRect{ 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) },
	aspectRatio(width / real(height)),
	frameIndex(0),
	rtvHeapHandleIncrementSize(0),
	fps(0.0)
{ }

bool D3D12HelloTriangle::init() {
	if (!loadPipeline()) {
		return false;
	}
	
	return loadAssets();
}

void D3D12HelloTriangle::deinit() {
	flush();
}

void D3D12HelloTriangle::update() {
	timeIt();
}

void D3D12HelloTriangle::render() {
	populateCommandList();

	ID3D12CommandList *const commandLists[] = { commandListsDirect[frameIndex].Get() };
	commandQueueDirect->ExecuteCommandLists(_countof(commandLists), commandLists);

	UINT syncInterval = Win32App::vSyncEnabled ? 1 : 0;
	UINT presentFlags = Win32App::tearingEnabled && !Win32App::vSyncEnabled ? DXGI_PRESENT_ALLOW_TEARING : 0;
	RETURN_ON_ERROR(swapChain->Present(syncInterval, presentFlags), ,"Failed to execute command list!\n");
	
	frameFenceValues[frameIndex] = signal();
	frameIndex = swapChain->GetCurrentBackBufferIndex();
	waitForFenceValue(frameFenceValues[frameIndex]);
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

	flush();

	for (int i = 0; i < frameCount; ++i) {
		backBuffers[i].Reset();
		frameFenceValues[i] = frameFenceValues[frameIndex];
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

void GetHardwareAdapter(
		IDXGIFactory1 *pFactory,
		IDXGIAdapter1 **ppAdapter,
		bool requestHighPerformanceAdapter
) {
	*ppAdapter = nullptr;

	ComPtr<IDXGIAdapter1> adapter;

	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
		for (
				UINT adapterIndex = 0;
				DXGI_ERROR_NOT_FOUND != factory6->EnumAdapterByGpuPreference(
					adapterIndex,
					requestHighPerformanceAdapter == true ? DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE : DXGI_GPU_PREFERENCE_UNSPECIFIED,
					IID_PPV_ARGS(&adapter)
				);
				++adapterIndex) {
			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
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

bool checkTearingSupport() {
	ComPtr<IDXGIFactory5> dxgiFactory;
	if (!SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)))) {
		return false;
	}

	bool allowTearing = false;
	if (FAILED(dxgiFactory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
		allowTearing = false;
	}

	return (allowTearing == true);
}

ComPtr<ID3D12CommandQueue> D3D12HelloTriangle::createCommandQueue(D3D12_COMMAND_LIST_TYPE type) {
	ComPtr<ID3D12CommandQueue> cmdQueue;
	D3D12_COMMAND_QUEUE_DESC cqDesc = {};
	cqDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	cqDesc.Type = type;

	RETURN_FALSE_ON_ERROR(
	device->CreateCommandQueue(&cqDesc, IID_PPV_ARGS(&cmdQueue)),
		"Failed to create command queue!\n"
	);

	return cmdQueue;
}

ComPtr<ID3D12CommandAllocator> D3D12HelloTriangle::createCommandAllocator(D3D12_COMMAND_LIST_TYPE type) {
	ComPtr<ID3D12CommandAllocator> cmdAllocator;
	RETURN_NULL_ON_ERROR(
		device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator)),
		"Failed to create command allocator!\n"
	);
	return cmdAllocator;
}


ComPtr<ID3D12GraphicsCommandList> D3D12HelloTriangle::createCommandList(ComPtr<ID3D12CommandAllocator> cmdAllocator, D3D12_COMMAND_LIST_TYPE type) {
	ComPtr<ID3D12GraphicsCommandList> cmdList;

	RETURN_FALSE_ON_ERROR(
		device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAllocator.Get(), nullptr, IID_PPV_ARGS(&cmdList)),
		"Failed to create command list!\n"
	);

	RETURN_FALSE_ON_ERROR(
		cmdList->Close(),
		"Failed to close the command list after creation!\n"
	);

	return cmdList;
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
	UINT createFactoryFlags = 0;
#if defined(D3D12_DEBUG)
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
#endif // defined(D3D12_DEBUG)

	RETURN_FALSE_ON_ERROR(
		CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)),
		"Error while creating DXGI Factory!\n"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	GetHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter, false);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
		"Failed to create device!\n"
	);

	/* Create command queue */
	commandQueueDirect = createCommandQueue(D3D12_COMMAND_LIST_TYPE_DIRECT);

	/* Create a swap chain */
	Win32App::tearingEnabled = checkTearingSupport();

	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	scDesc.Width = width;
	scDesc.Height = height;
	scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.Stereo = FALSE;
	// NOTE: if multisampling should be enabled the bitblt transfer swap method should be used
	scDesc.SampleDesc = { 1, 0 }; // Not using multi-sampling.
	scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.BufferCount = frameCount;
	scDesc.Scaling = DXGI_SCALING_STRETCH;
	scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	scDesc.Flags = Win32App::tearingEnabled ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
	
	ComPtr<IDXGISwapChain1> swapChainPlaceholder;
	RETURN_FALSE_ON_ERROR(
		dxgiFactory->CreateSwapChainForHwnd(
			commandQueueDirect.Get(), // Swap chain needs the queue so that it can force a flush on it.
			Win32App::getWindow(),
			&scDesc,
			NULL /*DXGI_SWAP_CHAIN_FULLSCREEN_DESC*/, // will be handled manually
			NULL /*pRestrictToOutput*/,
			&swapChainPlaceholder
		),
		"Failed to create swap chain!\n"
	);

	RETURN_FALSE_ON_ERROR(
		dxgiFactory->MakeWindowAssociation(Win32App::getWindow(), DXGI_MWA_NO_ALT_ENTER),
		"Failed to make window association NO_ALT_ENTER\n"
	);
	
	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!\n"
	);

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	/* Create a descriptor heap for RTVs */
	D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = { };
	rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
	rtvHeapDesc.NumDescriptors = frameCount;
	rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	// TODO: 0 since using only one device. Look into that.
	rtvHeapDesc.NodeMask = 0;

	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)),
		"Failed to create descriptor heap!\n"
	);

	/* Create render target view */
	rtvHeapHandleIncrementSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	if (!updateRenderTargetViews()) {
		return false;
	}

	/* Lastly, create a command list for each back buffer and their responding allocators */
	for (int i = 0; i < frameCount; ++i) {
		commandAllocatorsDirect[i] = createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT);
		commandListsDirect[i] = createCommandList(commandAllocatorsDirect[i], D3D12_COMMAND_LIST_TYPE_DIRECT);
	}

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
	// TODO: do not compile shaders runtime
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

	flush();

	return true;
}

void D3D12HelloTriangle::populateCommandList() {
	RETURN_ON_ERROR(commandAllocatorsDirect[frameIndex]->Reset(), , "Failed to reset the command allocator!\n");
	RETURN_ON_ERROR(commandListsDirect[frameIndex]->Reset(commandAllocatorsDirect[frameIndex].Get(), pipelineState.Get()), , "Failed to reset the command list!\n");

	commandListsDirect[frameIndex]->SetPipelineState(pipelineState.Get());
	commandListsDirect[frameIndex]->SetGraphicsRootSignature(rootSignature.Get());
	commandListsDirect[frameIndex]->RSSetViewports(1, &viewport);
	commandListsDirect[frameIndex]->RSSetScissorRects(1, &scissorRect);

	CD3DX12_RESOURCE_BARRIER resBarrierPresetToRT = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_PRESENT,
		D3D12_RESOURCE_STATE_RENDER_TARGET
	);
	commandListsDirect[frameIndex]->ResourceBarrier(1, &resBarrierPresetToRT);

	CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvHeapHandleIncrementSize);
	commandListsDirect[frameIndex]->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

	const float clearColor[] = { 0.2f, 0.2f, 0.8f, 1.f };
	commandListsDirect[frameIndex]->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
	
	commandListsDirect[frameIndex]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandListsDirect[frameIndex]->IASetVertexBuffers(0, 1, &vertexBufferView);
	commandListsDirect[frameIndex]->DrawInstanced(3, 1, 0, 0);

	CD3DX12_RESOURCE_BARRIER resBarrierRTtoPresent = CD3DX12_RESOURCE_BARRIER::Transition(
		backBuffers[frameIndex].Get(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	commandListsDirect[frameIndex]->ResourceBarrier(1, &resBarrierRTtoPresent);

	commandListsDirect[frameIndex]->Close();
}

UINT64 D3D12HelloTriangle::signal() {
	UINT64 fenceVal = ++fenceValue;
	RETURN_ON_ERROR(commandQueueDirect->Signal(fence.Get(), fenceVal), 0, "Failed to signal command queue!\n");

	return fenceVal;
}

void D3D12HelloTriangle::waitForFenceValue(UINT64 fenceVal) {
	while (fence->GetCompletedValue() < fenceVal) {
		RETURN_ON_ERROR(
			fence->SetEventOnCompletion(fenceVal, fenceEvent), ,
			"Failed to set fence event on completion!\n"
		);
		WaitForSingleObject(fenceEvent, INFINITE);
	}
}

void D3D12HelloTriangle::flush() {
	UINT64 fenceVal = signal();
	waitForFenceValue(fenceVal);
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
