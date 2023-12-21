#include "device.h"

#include "framework/app.h"
#include "utils/defines.h"

#include "imgui.h"
#include "imgui/backends/imgui_impl_dx12.h"
#include "imgui/backends/imgui_impl_glfw.h"

namespace Dar {

void getHardwareAdapter(
	IDXGIFactory1 *pFactory,
	IDXGIAdapter1 **ppAdapter
) {
	*ppAdapter = nullptr;

	ComPtr<IDXGIAdapter1> adapter;

	ComPtr<IDXGIFactory6> factory6;
	if (SUCCEEDED(pFactory->QueryInterface(IID_PPV_ARGS(&factory6)))) {
		for (
			UINT adapterIndex = 0;
			DXGI_ERROR_NOT_FOUND != factory6->EnumAdapterByGpuPreference(
				adapterIndex,
				DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
				IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())
			);
			++adapterIndex) {
			if (adapter == nullptr) {
				continue;
			}

			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1(&desc);

			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
				continue;
			}

			// Check to see whether the adapter supports Direct3D 12, but don't create the
			// actual device yet.
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
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
			if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device), nullptr))) {
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

Device::Device() : commandQueueDirect(D3D12_COMMAND_LIST_TYPE_DIRECT) {}

bool Device::init() {
#if defined(DAR_DEBUG)
	/* Enable debug layer */
	{
		ComPtr<ID3D12Debug> debugLayer;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
			debugLayer->EnableDebugLayer();
		}
	}
#endif // defined(DAR_DEBUG)

	/* Create the device */
	ComPtr<IDXGIFactory4> dxgiFactory;
	UINT createFactoryFlags = 0;
	#if defined(DAR_DEBUG)
	createFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
	#endif // defined(DAR_DEBUG)

	RETURN_FALSE_ON_ERROR(
		CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)),
		"Error while creating DXGI Factory!"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	getHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)),
		"Failed to create device!"
	);

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_6 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.6!"
	);

	if (shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_6) {
		fprintf(stderr, "Shader model 6.6 not supported!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS options = { };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)),
		"Failed to check features support!"
	);

	if (options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
		fprintf(stderr, "GPU does not support resource binding tier 3!");
		return false;
	}

	D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7 = { };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7)),
		"Failed to check features support!"
	);

	// Note: we don't do mesh shading currently, yet we will, so we want to support it.
	/*if (options7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
		fprintf(stderr, "Mesh shading is not supported!");
		return false;
	}*/

	/* Create command queue */
	commandQueueDirect.init(device);

	/* Create a swap chain */
	allowTearing = checkTearingSupport();
	
	backbuffer.init(dxgiFactory, commandQueueDirect, allowTearing);

	if (!initImGui()) {
		LOG(Error, "Failed to initialize ImGui!");
		return false;
	}

	return true;
}

void Device::deinit() {
	device.Reset();

	commandQueueDirect.deinit();

	backbuffer.deinit();

#ifdef DAR_DEBUG
	HMODULE dxgiModule = GetModuleHandleA("Dxgidebug.dll");
	if (!dxgiModule) {
		return;
	}

	using DXGIGetDebugInterfaceProc = HRESULT(*)(REFIID riid, void **ppDebug);

	auto dxgiGetDebugInterface = (DXGIGetDebugInterfaceProc)GetProcAddress(dxgiModule, "DXGIGetDebugInterface");

	if (!dxgiGetDebugInterface) {
		return;
	}

	ComPtr<IDXGIDebug> debugLayer;
	if (SUCCEEDED(dxgiGetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
		debugLayer->ReportLiveObjects(DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_IGNORE_INTERNAL | DXGI_DEBUG_RLO_SUMMARY));
	}
#endif // DAR_DEBUG
}

ComPtr<ID3D12Device> Device::getDevice() const {
	return device;
}

ID3D12Device* Device::getDevicePtr() const {
	return device.Get();
}

void Device::registerBackBuffersInResourceManager() {
	backbuffer.registerInResourceManager();
}

bool Device::resizeBackBuffers() {
	if (!backbuffer.resize()) {
		return false;
	}

	deinitImGui();
	initImGui();

	return true;
}

bool Device::initImGui() {
	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 1;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	createSRVHeap(srvHeapDesc, imguiSRVHeap);

	auto *app = getApp();

	ImGuiIO &io = ImGui::GetIO();
	io.DisplaySize.x = float(app->getWidth());
	io.DisplaySize.y = float(app->getHeight());

	ImGui_ImplGlfw_InitForOther(app->getGLFWWindow(), true);
	ImGui_ImplDX12_Init(
		getDevice().Get(),
		FRAME_COUNT,
		DXGI_FORMAT_R8G8B8A8_UNORM,
		imguiSRVHeap.Get(),
		imguiSRVHeap->GetCPUDescriptorHandleForHeapStart(),
		imguiSRVHeap->GetGPUDescriptorHandleForHeapStart()
	);

	ImGui_ImplDX12_CreateDeviceObjects();

	imGuiShutdown = false;

	return true;
}

bool Device::deinitImGui() {
	ImGui_ImplDX12_InvalidateDeviceObjects();
	if (!imGuiShutdown) {
		ImGui_ImplDX12_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		imGuiShutdown = true;
	}
	imguiSRVHeap.Reset();

	return true;
}

} // namespace Dar