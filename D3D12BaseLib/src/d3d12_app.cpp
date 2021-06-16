#include "d3d12_app.h"

#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>

#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include "d3dx12.h"
#include "d3d12_defines.h"

#include <bitset>
#include <cstdio>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 4; }

extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8".\\"; }

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

////////////
// D3D12App
////////////
D3D12App::D3D12App(UINT width, UINT height, const char *windowTitle) :
	commandQueueDirect(D3D12_COMMAND_LIST_TYPE_DIRECT),
	commandQueueCopy(D3D12_COMMAND_LIST_TYPE_COPY),
	frameIndex(0),
	width(width),
	height(height),
	window(nullptr),
	windowRect{},
	vSyncEnabled(false),
	allowTearing(false),
	fullscreen(false) {
	strncpy(title, windowTitle, strlen(windowTitle) + 1);
	title[strlen(windowTitle)] = '\0';
}

D3D12App::~D3D12App() { }

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
		"Error while creating DXGI Factory!\n"
	);

	ComPtr<IDXGIAdapter1> hardwareAdapter;
	GetHardwareAdapter(dxgiFactory.Get(), &hardwareAdapter, false);

	RETURN_FALSE_ON_ERROR(D3D12CreateDevice(
		hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
		"Failed to create device!\n"
	);

	D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_6 };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
		"Device does not support shader model 6.6!\n"
	);

	D3D12_FEATURE_DATA_D3D12_OPTIONS options = { };
	RETURN_FALSE_ON_ERROR(
		device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)),
		"Failed to check features support!\n"
	);

	if (options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
		return false;
	}

	/* Create command queue */
	commandQueueDirect.init(device);
	commandQueueCopy.init(device);

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
		"Failed to create swap chain!\n"
	);

	RETURN_FALSE_ON_ERROR(
		dxgiFactory->MakeWindowAssociation(window, DXGI_MWA_NO_ALT_ENTER),
		"Failed to make window association NO_ALT_ENTER\n"
	);

	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!\n"
	);

	frameIndex = swapChain->GetCurrentBackBufferIndex();

	// cache root signature's feature version
	rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
	if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &rootSignatureFeatureData, sizeof(rootSignatureFeatureData)))) {
		rootSignatureFeatureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
	}

	return true;
}

void D3D12App::toggleFullscreen() {
	fullscreen = !fullscreen;

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
		SetWindowPos(hWnd, HWND_TOP,
				monitorInfo.rcMonitor.left,
				monitorInfo.rcMonitor.top,
				monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left,
				monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top,
				SWP_FRAMECHANGED | SWP_NOACTIVATE);

		::ShowWindow(hWnd, SW_MAXIMIZE);
	} else { // restore previous dimensions
		::SetWindowLong(hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);

		::SetWindowPos(hWnd, HWND_NOTOPMOST,
				windowRect.left,
				windowRect.top,
				windowRect.right - windowRect.left,
				windowRect.bottom - windowRect.top,
				SWP_FRAMECHANGED | SWP_NOACTIVATE);

		::ShowWindow(hWnd, SW_NORMAL);
	}
}

HWND D3D12App::getWindow() const {
	return window;
}

void D3D12App::flush() {
	commandQueueCopy.flush();
	commandQueueDirect.flush();
}

int D3D12App::run() {
	app = this; // save global state for glfw callbacks

	while (!glfwWindowShouldClose(glfwWindow)) {
		processKeyboardInput(glfwWindow);

		update();
		render();

		glfwPollEvents();
	}

	deinit();
	glfwTerminate();

	return 0;
}
