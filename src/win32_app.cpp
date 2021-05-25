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
D3D12App* Win32App::app = nullptr;

////////////
// GLFW
////////////
void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
	Win32App::getApp()->resize(width, height);
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
	GLFWwindow *glfwWindow = glfwCreateWindow(app->getWidth(), app->getHeight(), app->getTitle(), nullptr, nullptr);
	if (glfwWindow == nullptr) {
		return -1;
	}

	glfwSetFramebufferSizeCallback(glfwWindow, framebufferSizeCallback);

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

D3D12App* Win32App::getApp() {
	return app;
}