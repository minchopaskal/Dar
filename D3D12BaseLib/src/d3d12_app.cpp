#include "d3d12_app.h"

#include "d3d12_command_list.h"
#include "d3d12_defines.h"
#include "d3d12_resource_manager.h"

#include "d3dx12.h"

#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>

#include "imgui.h"
#include "imgui/backends/imgui_impl_dx12.h"
#include "imgui/backends/imgui_impl_glfw.h"

#include <d3dcompiler.h>
#include <dxgi1_6.h>

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 4; }

extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8".\\D3D12\\"; }

/////////////////////////////////
// Global state
/////////////////////////////////
GLFWwindow* glfwWindow = nullptr;
D3D12App* app = nullptr;

////////////
// GLFW
////////////
void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
	app->onResize(width, height);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key < GLFW_KEY_0 || key > GLFW_KEY_Z) {
		return;
	}
	
	app->keyPressed[key] = !(action == GLFW_RELEASE);
	app->keyRepeated[key] = (action == GLFW_REPEAT);

	// TODO: mapping keys to engine actions
	app->onKeyboardInput(key, action);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	app->onMouseScroll(xoffset, yoffset);
}

void windowPosCallback(GLFWwindow *window, int xpos, int ypos) {
	app->onWindowPosChange(xpos, ypos);
}

void processKeyboardInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, 1);
	}
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	app->onMouseMove(xpos, ypos);
}

////////////
// D3D12App
////////////
D3D12App::D3D12App(UINT width, UINT height, const char *windowTitle) :
	commandQueueDirect(D3D12_COMMAND_LIST_TYPE_DIRECT),
	resManager(nullptr),
	frameIndex(0),
	width(width),
	height(height),
	abort(false),
	window(nullptr),
	windowRect{},
	vSyncEnabled(false),
	allowTearing(false),
	fullscreen(false),
	useImGui(false),
	imGuiShutdown(true) {
	strncpy(title, windowTitle, strlen(windowTitle) + 1);
	title[strlen(windowTitle)] = '\0';
}

D3D12App::~D3D12App() {}

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
				DXGI_ERROR_NOT_FOUND != factory6->EnumAdapters1(
					adapterIndex,
					&adapter
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

int D3D12App::init() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (glfwWindow == nullptr) {
		return 1;
	}

	glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
	glfwSetKeyCallback(glfwWindow, keyCallback);
	glfwSetScrollCallback(glfwWindow, scrollCallback);
	glfwSetWindowPosCallback(glfwWindow, windowPosCallback);
	glfwSetCursorPosCallback(glfwWindow, cursorPositionCallback);

	glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	window = glfwGetWin32Window(glfwWindow);

#if defined(D3D12_DEBUG)
/* Enable debug layer */
	{
		ComPtr<ID3D12Debug> debugLayer;
		if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugLayer)))) {
			debugLayer->EnableDebugLayer();
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
		"Error while creating DXGI Factory!"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	GetHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter, false);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_1, IID_PPV_ARGS(&device)),
		"Failed to create device!"
	);

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_5 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.5!"
	);

	if (shaderModel.HighestShaderModel != D3D_SHADER_MODEL_6_5) {
		fprintf(stderr, "Shader model 6.5 not supported!");
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
	scDesc.Flags = allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	ComPtr<IDXGISwapChain1> swapChainPlaceholder;
	RETURN_FALSE_ON_ERROR(
		dxgiFactory->CreateSwapChainForHwnd(
		commandQueueDirect.getCommandQueue().Get(), // Swap chain needs the queue so that it can force a flush on it.
		window,
		&scDesc,
		NULL /*DXGI_SWAP_CHAIN_FULLSCREEN_DESC*/, // will be handled manually
		NULL /*pRestrictToOutput*/,
		&swapChainPlaceholder
	),
		"Failed to create swap chain!"
	);

	RETURN_FALSE_ON_ERROR(
		dxgiFactory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER),
		"Failed to make window association NO_ALT_ENTER"
	);

	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!"
	);

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// cache root signature's feature version
	rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
	if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignatureFeatureData, sizeof(rootSignatureFeatureData)))) {
		rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	// TODO: threads
	initResourceManager(device, 1);
	resManager = &getResourceManager();

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
	srvHeapDesc.NumDescriptors = 1;
	srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	RETURN_FALSE_ON_ERROR(
		device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&imguiSRVHeap)),
		"Failed to create ImGui's SRV descriptor heap!"
	);

	ImGuiIO &io = ImGui::GetIO();
	io.DisplaySize.x = float(width);
	io.DisplaySize.y = float(height);

	ImGui_ImplGlfw_InitForOther(glfwWindow, true);
	ImGui_ImplDX12_Init(
		device.Get(),
		frameCount,
		DXGI_FORMAT_R8G8B8A8_UNORM, 
		imguiSRVHeap.Get(),
		imguiSRVHeap->GetCPUDescriptorHandleForHeapStart(),
		imguiSRVHeap->GetGPUDescriptorHandleForHeapStart()
	);

	ImGui_ImplDX12_CreateDeviceObjects();

	imGuiShutdown = false;

	return true;
}

void D3D12App::deinit() {
	deinitResourceManager();
	resManager = nullptr;
	ImGui_ImplDX12_InvalidateDeviceObjects();
	if (!imGuiShutdown) {
		ImGui_ImplDX12_Shutdown();
		imGuiShutdown = true;
	}
}

void D3D12App::setUseImGui() {
	useImGui = true;
}

void D3D12App::renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle) {
	if (!useImGui) {
		return;
	}

	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	drawUI();

	ImGui::Render();

	cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
	cmdList->SetDescriptorHeaps(1, imguiSRVHeap.GetAddressOf());
	ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList.get());
}

int D3D12App::getWidth() const {
	return width;
}

int D3D12App::getHeight() const {
	return height;
}

ButtonState D3D12App::query(char key) {
	key = isalpha(key) ? toupper(key) : key;

	if (key < GLFW_KEY_0 || key > GLFW_KEY_Z) {
		return { false, false };
	}

	ButtonState res;
	res.pressed = keyPressed[key];
	res.repeated = keyRepeated[key];

	return res;
}

void D3D12App::toggleFullscreen() {
	fullscreen = !fullscreen;

	UINT width;
	UINT height;

	// TODO: do this in GLFW to get rid of this ugliness
	HWND hWnd = window;
	if (fullscreen) {
		GetWindowRect(hWnd, &windowRect);
		UINT windowStyle = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);

		SetWindowLongW(hWnd, GWL_STYLE, windowStyle);
		HMONITOR hMonitor = ::MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
		MONITORINFOEX monitorInfo = {};
		monitorInfo.cbSize = sizeof(MONITORINFOEX);
		GetMonitorInfo(hMonitor, &monitorInfo);
		width = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
		height = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
		SetWindowPos(hWnd, HWND_TOP,
				monitorInfo.rcMonitor.left,
				monitorInfo.rcMonitor.top,
				width,
				height,
				SWP_FRAMECHANGED | SWP_NOACTIVATE);

		app->onResize(width, height);
		::ShowWindow(hWnd, SW_MAXIMIZE);
	} else { // restore previous dimensions
		::SetWindowLong(hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);

		width = windowRect.right - windowRect.left;
		height = windowRect.bottom - windowRect.top;
		::SetWindowPos(hWnd, HWND_NOTOPMOST,
				windowRect.left,
				windowRect.top,
				width,
				height,
				SWP_FRAMECHANGED | SWP_NOACTIVATE);

		app->onResize(width, height);
		::ShowWindow(hWnd, SW_NORMAL);
	}

}

HWND D3D12App::getWindow() const {
	return window;
}

void D3D12App::flush() {
	commandQueueDirect.flush();
	if (resManager) {
		resManager->flush();
	}
}

int D3D12App::run() {
	app = this; // save global state for glfw callbacks

	while (!abort && !glfwWindowShouldClose(glfwWindow)) {
		processKeyboardInput(glfwWindow);

		beginFrame();

		update();

		render();

		endFrame();
		resManager->endFrame();

		glfwPollEvents();
	}

	deinit();
	glfwTerminate();

	return 0;
}
