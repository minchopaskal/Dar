#include "win32_app.h"

#include "d3d12_app.h"

#include <glfw/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <glfw/glfw3native.h>

#include <cstdio>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>

HWND Win32App::window = nullptr;
GLFWwindow* Win32App::glfwWindow = nullptr;
D3D12App* Win32App::app = nullptr;
RECT Win32App::windowRect = { };
bool Win32App::fullscreen = false;
std::bitset<Win32App::keysCount> Win32App::keyPressed = {};
std::bitset<Win32App::keysCount> Win32App::keyRepeated = {};
bool Win32App::vSyncEnabled = true;
bool Win32App::tearingEnabled = false;

////////////
// GLFW
////////////
void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
	D3D12App *app = Win32App::getD3D12App();
	app->resize(width, height);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key >= GLFW_KEY_SPACE && key <= GLFW_KEY_Z) {
		Win32App::keyPressed[key] = !(action == GLFW_RELEASE);
		Win32App::keyRepeated[key] = (action == GLFW_REPEAT);
	}

	// TODO: mapping keys to engine actions
	if (Win32App::keyPressed[GLFW_KEY_F] && !Win32App::keyRepeated[GLFW_KEY_F]) {
		Win32App::toggleFullscreen();
	}

	Win32App::vSyncEnabled = Win32App::keyPressed[GLFW_KEY_V];
}

void processKeyboardInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, 1);
	}
}

////////////
// Win32App
////////////
int Win32App::Run(D3D12App *d3d12App) {
	app = d3d12App;

	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindow = glfwCreateWindow(app->getWidth(), app->getHeight(), app->getTitle(), nullptr, nullptr);
	if (glfwWindow == nullptr) {
		return -1;
	}

	glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);
	glfwSetKeyCallback(glfwWindow, keyCallback);

	window = glfwGetWin32Window(glfwWindow);

	app->init();

	// Resize window for the first time
	framebufferSizeCallback(glfwWindow, app->getWidth(), app->getHeight());

	while (!glfwWindowShouldClose(glfwWindow)) {
		processKeyboardInput(glfwWindow);

		app->update();
		app->render();

		glfwPollEvents();
	}

	app->deinit();
	glfwTerminate();
}

HWND Win32App::getWindow() {
	return window;
}

D3D12App* Win32App::getD3D12App() {
	return app;
}

void Win32App::toggleFullscreen() {
	fullscreen = !fullscreen;

	// TODO: do this in GLFW to get rid of this ugliness
	HWND hWnd = getWindow();
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