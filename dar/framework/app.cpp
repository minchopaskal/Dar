#include "framework/app.h"

#include "async/job_system.h"
#include "d3d12/command_list.h"
#include "d3d12/resource_manager.h"
#include "utils/defines.h"
#include "utils/profile.h"

#include "reslib/resource_library.h"

#include "d3dx12.h"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>

extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 4; }

extern "C" { __declspec(dllexport) extern const char8_t* D3D12SDKPath = u8".\\D3D12\\"; }

/////////////////////////////////
// Global state
/////////////////////////////////
GLFWwindow* glfwWindow = nullptr;
static Dar::App* g_App = nullptr;

namespace Dar {

////////////
// GLFW
////////////
void framebufferSizeCallback(GLFWwindow*, int width, int height) {
	g_App->onResize(width, height);
}

void keyCallback(GLFWwindow*, int key, int /*scancode*/, int action, int /*mods*/) {
	if (key == GLFW_KEY_UNKNOWN) {
		LOG(Warning, "Unknown key pressed!");
		return;
	}

	g_App->keyPressed[key] = (action == GLFW_PRESS);
	g_App->keyReleased[key] = (action == GLFW_RELEASE);
	g_App->keyRepeated[key] = (action == GLFW_REPEAT);

	// TODO: mapping keys to engine actions
	g_App->onKeyboardInput(key, action);
}

void scrollCallback(GLFWwindow*, double xoffset, double yoffset) {
	g_App->onMouseScroll(xoffset, yoffset);
}

void windowPosCallback(GLFWwindow*, int xpos, int ypos) {
	g_App->onWindowPosChange(xpos, ypos);
}

void cursorPositionCallback(GLFWwindow*, double xpos, double ypos) {
	g_App->onMouseMove(xpos, ypos);
}

void windowCloseCallback(GLFWwindow*) {
	g_App->onWindowClose();
}

void mouseButtonCallback(GLFWwindow*, int button, int action, int mods) {
	g_App->onMouseButton(button, action, mods);
}

App *getApp() {
	return g_App;
}

////////////
// App
////////////
App::App(UINT width, UINT height, const char *windowTitle) :
	resManager(nullptr),
	width(width),
	height(height),
	abort(false),
	window(nullptr),
	windowRect{},
	fullscreen(false)
{
	strncpy(title, windowTitle, strlen(windowTitle) + 1);
	title[strlen(windowTitle)] = '\0';
	setNumThreads(-1);
}

App::~App() {
	
}

bool App::init() {
	LOG(Info, "App::init");

	g_App = this; // save global state for glfw callbacks

	JobSystem::init(numThreads);

	initRes.app = this;
	// GLFW needs to be loaded in the main thread so
	// it's outside of the initJob job, which in its turn
	// must not to be loaded on the main thread because the main loop is
	// executed there and this would prevent asynchronous loading.
	auto initJobLambda = [](void *param) {
		auto initRes = reinterpret_cast<InitRes*>(param);
		auto app = initRes->app;
		auto& res = initRes->res;
		res = app->initJob();

		app->initImpl();
	};
	JobSystem::JobDecl initJobDecl = {};
	initJobDecl.f = initJobLambda;
	initJobDecl.param = &initRes;
	JobSystem::kickJobs(&initJobDecl, 1, &initJobFence, JobSystem::JobType::Windows);

	return true;
}

bool App::initJob() {
	LOG(Info, "App::initJob");

	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindow = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (glfwWindow == nullptr) {
		return false;
	}

	glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
	glfwSetKeyCallback(glfwWindow, keyCallback);
	glfwSetScrollCallback(glfwWindow, scrollCallback);
	glfwSetWindowPosCallback(glfwWindow, windowPosCallback);
	glfwSetCursorPosCallback(glfwWindow, cursorPositionCallback);
	glfwSetWindowCloseCallback(glfwWindow, windowCloseCallback);
	glfwSetMouseButtonCallback(glfwWindow, mouseButtonCallback);

	window = glfwGetWin32Window(glfwWindow);

	initResourceLibrary();
	
	device.init();

	initResourceManager(device.getDevice(), JobSystem::getNumThreads());
	resManager = &getResourceManager();

	device.registerBackBuffersInResourceManager();

	LOG(Info, "App::initJob SUCCESS");

	return true;
}

void App::deinit() {
	LOG(Info, "App::deinit");

	deinitResourceLibrary();
	deinitResourceManager();
	resManager = nullptr;

	LOG(Info, "App::deinit SUCCESS");
}

int App::getWidth() const {
	return width;
}

int App::getHeight() const {
	return height;
}

ButtonState App::query(int key) {
	ButtonState res;
	res.pressed = keyPressed[key] || keyRepeated[key];
	res.repeated = keyRepeated[key];
	res.released = keyReleased[key];
	res.justPressed = keyPressed[key];

	return res;
}

bool App::queryPressed(int key) {
	return keyPressed[key];
}

bool App::queryReleased(int key) {
	return keyReleased[key];
}

GLFWwindow *App::getGLFWWindow() const {
	return glfwWindow;
}

void App::setNumThreads(int nt) {
	numThreads = nt;
}

void App::toggleFullscreen() {
	fullscreen = !fullscreen;

	UINT w;
	UINT h;

	// TODO: do this in GLFW to get rid of this ugliness
	HWND hWnd = window;
	if (fullscreen) {
		::GetWindowRect(hWnd, &windowRect);
		::GetClientRect(hWnd, &clientRect);
		UINT windowStyle = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);

		::SetWindowLongW(hWnd, GWL_STYLE, windowStyle);
		HMONITOR hMonitor = ::MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
		MONITORINFOEX monitorInfo = {};
		monitorInfo.cbSize = sizeof(MONITORINFOEX);
		GetMonitorInfo(hMonitor, &monitorInfo);
		w = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
		h = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
		::SetWindowPos(hWnd, HWND_TOP,
			monitorInfo.rcMonitor.left,
			monitorInfo.rcMonitor.top,
			w,
			h,
			SWP_FRAMECHANGED | SWP_NOACTIVATE);

		g_App->onResize(w, h);
		::ShowWindow(hWnd, SW_MAXIMIZE);
	} else { // restore previous dimensions
		::SetWindowLong(hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);

		w = windowRect.right - windowRect.left;
		h = windowRect.bottom - windowRect.top;
		::SetWindowPos(hWnd, HWND_NOTOPMOST,
			windowRect.left,
			windowRect.top,
			w,
			h,
			SWP_FRAMECHANGED | SWP_NOACTIVATE);


		w = clientRect.right - clientRect.left;
		h = clientRect.bottom - clientRect.top;
		g_App->onResize(w, h);
		::ShowWindow(hWnd, SW_NORMAL);
	}
}

HWND App::getWindow() const {
	return window;
}

void App::flush() {
	device.flushCommandQueue();
	if (resManager) {
		resManager->flush();
	}
}

int App::run() {
	LOG(Info, "App::run");

	auto mainLoop = [](void *param) {
		App *app = reinterpret_cast<App*>(param);
		
		JobSystem::waitFenceAndFree(app->initJobFence);
		LOG(Info, "App::init SUCCESS");

		app->mainLoopJob();
	};

	JobSystem::JobDecl mainLoopJobDecl = {};
	mainLoopJobDecl.f = mainLoop;
	mainLoopJobDecl.param = this;

	JobSystem::kickJobs(&mainLoopJobDecl, 1, nullptr, JobSystem::JobType::Windows);

	JobSystem::waitForAll();

	return 0;
}

void App::timeIt() {
	using std::chrono::duration_cast;
	constexpr double seconds_in_nanosecond = 1e-9;

	if (frameCount == 0) {
		++frameCount;
		currentTime = HRC::now();
	}

	const HRC::time_point t1 = HRC::now();
	deltaTime = static_cast<double>((t1 - currentTime).count()) * seconds_in_nanosecond;
	elapsedTime += deltaTime;
	totalTime += deltaTime;

	++frameCount;
	currentTime = t1;

	if (elapsedTime > 1.0) {
		fps = static_cast<double>(frameCount) / elapsedTime;
		frameTime = deltaTime * 1000.;

#if defined(DAR_DEBUG)
		char buffer[512];
		sprintf_s(buffer, "FPS: %.2f\n", fps);
		OutputDebugString(buffer);
#endif // defined(DAR_DEBUG)

		frameCount = 0;
		elapsedTime = 0.0;
	}
}

int App::mainLoopJob() {
	LOG(Info, "App::mainLoop");

	if (!initRes.res) {
		// TODO: write a defer macro
		Dar::JobSystem::stop();

		LOG(Error, "Failure during initialization!");
		return 1;
	}

	while (!abort && !glfwWindowShouldClose(glfwWindow)) {
		DAR_OPTICK_FRAME("Frame");

		timeIt();

		beginFrame();

		update();

		endFrame();

		resManager->endFrame();

		glfwPollEvents();
	}

	deinit();
	glfwTerminate();

	Dar::JobSystem::stop();

	return 0;
}

} // namespace Dar