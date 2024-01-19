#pragma once

#include "d3d12/includes.h"

#include "d3d12/command_queue.h"

#include "renderer.h"

#include "framework/input_query.h"

#include "imgui/imgui.h"

#include <chrono>
using HRC = std::chrono::high_resolution_clock;

struct GLFWwindow;
struct ResourceManager;
struct CommandList;
namespace Dar {

namespace JobSystem {
struct Fence;
}

struct App : public IKeyboardInputQuery {
	/// @brieff App constructor
	/// @param width Width for rendering.
	/// @param height Height for rendering
	/// @param windowTitle Title of the window
	/// @param numThreads Num threads to run the system on. Default -1 means use all system threads.
	App(UINT width, UINT height, const char *windowTitle, int numThreads = -1);
	virtual ~App();

	/// The main loop.
	int run();

	/// Initialization work. Should be called before run().
	/// @return false on failure, true on success
	bool init();

	/// Return width of window
	int getWidth() const;

	/// Return height of window
	int getHeight() const;

	// Inherited by IKeyboardInputQuery
	ButtonState query(int key) override;

	bool queryPressed(int key) override;

	bool queryReleased(int key) override;

	/// Get pointer to the window.
	HWND getWindow() const;

	GLFWwindow *getGLFWWindow() const;

	/// Optional. It is advised that all ImGui draw calls go here, unless it's inconvinient.
	/// Called during renderUI().
	virtual void drawUI() {};

	double getDeltaTime() const {
		return deltaTime;
	}

protected:
	void setNumThreads(int numThreads);

	/// Toggle between windowed/fullscreen
	void toggleFullscreen();

	/// Flush any command queues' work.
	/// NB doesn't flush the resource manager's copy Queue.
	virtual void flush();

	/// Deinitialize the app.
	virtual void deinit() = 0;

	/// Called before update()
	virtual void beginFrame() {
		backbufferRenderedThisFrame = false;
	}

	/// Any update work should go here. Update is called after beginFrame() and before endFrame()
	virtual void update() = 0;

	/// Called after render()
	virtual void endFrame() {}

	/// Any initialization work should go here. Called inside init().
	/// @return false on failure, true on success
	virtual bool initImpl() = 0;

	double getFPS() const {
		return fps;
	}

	double getFrameTime() const {
		return frameTime;
	}

	double getTotalTime() const {
		return totalTime;
	}

	void quit() const {
		glfwSetWindowShouldClose(getGLFWWindow(), true);
	}

	virtual void onResize(const unsigned int w, const unsigned int h) = 0;
	virtual void onKeyboardInput(int /*key*/, int /*action */) {};
	virtual void onMouseScroll(double /*xOffset*/, double /*yOffset*/) {};
	virtual void onMouseMove(double /*xOffset*/, double /*yOffset*/) {};
	virtual void onWindowPosChange(int /*xPos*/, int /*yPos*/) {}
	virtual void onWindowClose() {}
	virtual void onMouseButton(int /*button*/, int /*action*/, int /*mods*/) {}

private:
	bool initJob();
	int mainLoopJob();

	void timeIt();

	void setBackbufferRendered() {
		dassertLog(backbufferRenderedThisFrame == false, "You can't render to the backbuffer more than once!");
		backbufferRenderedThisFrame = true;
	}

protected:
	Device device; ///< D3D12 Device

	ResourceManager *resManager; ///< Pointer to the resource manager.

	// TODO: get this out of here
	// Input
	static const int keysCount = GLFW_KEY_LAST;
	Bitset<keysCount> keyPressed;
	Bitset<keysCount> keyRepeated;
	Bitset<keysCount> keyReleased;

	char title[256]; ///< Title of the window
	UINT width, height; ///< Dimensions of the window

	Atomic<int> abort; ///< Abort the main loop if true.

private:
	struct InitRes {
		App* app;
		bool res = false;
	} initRes;

	// timing
	double fps = 0.0;
	double frameTime = 0.0;
	double totalTime = 0.0;
	double deltaTime = 0.0;
	double elapsedTime = 0.0;
	UINT64 frameCount = 0;
	HRC::time_point currentTime;

	// job system
	JobSystem::Fence *initJobFence = nullptr;
	int numThreads;

	HWND window; ///< Pointer to the win32 window abstraction
	RECT windowRect; ///< Window rectangle. Not to be confused with scissor rect
	RECT clientRect; ///< Window rectangle of client area.
	bool fullscreen; ///< Flag indicating whether or not the applicaion is in fullscreen.

	/// Exclusively debug flag, indicating whether we have rendered to the backbuffer already this frame.
	bool backbufferRenderedThisFrame = false;

	// GLFW callbacks
	friend void framebufferSizeCallback(GLFWwindow *window, int width, int height);
	friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	friend void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
	friend void windowPosCallback(GLFWwindow *window, int xpos, int ypos);
	friend void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
	friend void windowCloseCallback(GLFWwindow *window);
	friend void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

	friend class Renderer;
};

App *getApp();

}
