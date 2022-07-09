#include "framework/app.h"

#include "async/job_system.h"
#include "d3d12/command_list.h"
#include "d3d12/resource_manager.h"
#include "utils/defines.h"
#include "utils/profile.h"

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
void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
	g_App->onResize(width, height);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
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

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	g_App->onMouseScroll(xoffset, yoffset);
}

void windowPosCallback(GLFWwindow *window, int xpos, int ypos) {
	g_App->onWindowPosChange(xpos, ypos);
}

void processKeyboardInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, 1);
	}
}

void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos) {
	g_App->onMouseMove(xpos, ypos);
}

void windowCloseCallback(GLFWwindow *window) {
	g_App->onWindowClose();
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
}

App::~App() {
	
}

int App::init() {
	g_App = this; // save global state for glfw callbacks

	JobSystem::init();

	auto initJobLambda = [](void *param) {
		App *app = reinterpret_cast<App*>(param);
		app->initJob();
	};

	JobSystem::JobDecl initJobDecl = {};
	initJobDecl.f = initJobLambda;
	initJobDecl.param = this;

	JobSystem::kickJobs(&initJobDecl, 1, &initJobFence, JobSystem::JobType::Windows);

	return 1;
}

int App::initJob() {
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

	window = glfwGetWin32Window(glfwWindow);

	renderer.init();

	// TODO: threads
	initResourceManager(renderer.getDevice(), 1);
	resManager = &getResourceManager();

	renderer.registerBackBuffersInResourceManager();

	return initImpl();
}

void App::deinit() {
	renderer.deinit();
	deinitResourceManager();
	resManager = nullptr;
}

void App::setUseImGui() {
	renderer.getSettings().useImGui = true;
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

void App::toggleFullscreen() {
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

		g_App->onResize(width, height);
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

		g_App->onResize(width, height);
		::ShowWindow(hWnd, SW_NORMAL);
	}
}

HWND App::getWindow() const {
	return window;
}

void App::flush() {
	renderer.flush();
	if (resManager) {
		resManager->flush();
	}
}

int App::run() {
	auto mainLoop = [](void *param) {
		App *app = reinterpret_cast<App*>(param);

		JobSystem::waitFence(app->initJobFence);

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
	while (!abort && !glfwWindowShouldClose(glfwWindow)) {
		DAR_OPTICK_FRAME("Frame");

		timeIt();

		processKeyboardInput(glfwWindow);

		FrameData &fd = getFrameData();
		fd.beginFrame(renderer);

		beginFrame();

		renderer.beginFrame();

		update();

		renderer.renderFrame(getFrameData());

		renderer.endFrame();

		endFrame();
		
		fd.endFrame(renderer);

		resManager->endFrame();

		glfwPollEvents();
	}

	deinit();
	glfwTerminate();

	Dar::JobSystem::stop();

	return 0;
}

} // namespace Dar