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
	App(UINT width, UINT height, const char *windowTitle);
	virtual ~App();

	/// The main loop.
	int run();

	/// Initialization work. Should be called before run().
	/// @return 0 on failure, 1 on success
	int init();

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

protected:
	/// Toggle between windowed/fullscreen
	void toggleFullscreen();

	/// Flush any command queues' work.
	/// NB doesn't flush the resource manager's copy Queue.
	void flush();

	/// Deinitialize the app.
	virtual void deinit() = 0;

	/// Called before update()
	virtual void beginFrame() {}

	/// Called after render()
	virtual void endFrame() {}

	/// Any update work should go here. Update is called before render() in the main loop.
	virtual void update() = 0;

	/// Any class that inherits app should return a valid frame data which will be used by the renderer.
	virtual Dar::FrameData& getFrameData() = 0;

	/// Any initialization work should go here. Called inside init().
	/// @return 0 on failure, 1 on success
	virtual int initImpl() = 0;

	/// Optional. Should be called during init of the derived class if one intends to draw UI via ImGui.
	/// Note: call before D3D12::init() in order to skip ImGui initialization.
	void setUseImGui();

	double getFPS() const {
		return fps;
	}

	double getDeltaTime() const {
		return deltaTime;
	}

	double getTotalTime() const {
		return totalTime;
	}

	virtual void onResize(const unsigned int w, const unsigned int h) = 0;
	virtual void onKeyboardInput(int key, int action) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset) = 0;
	virtual void onMouseMove(double xOffset, double yOffset) {};
	virtual void onWindowPosChange(int xPos, int yPos) {}
	virtual void onWindowClose() {}

private:
	int initJob();
	int mainLoopJob();

	void timeIt();

protected:
	Renderer renderer;

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
	// timing
	double fps = 0.0;
	double totalTime = 0.0;
	double deltaTime = 0.0;
	double elapsedTime = 0.0;
	UINT64 frameCount = 0;
	HRC::time_point currentTime;

	JobSystem::Fence *initJobFence = nullptr;

	HWND window; ///< Pointer to the win32 window abstraction
	RECT windowRect; ///< Window rectangle. Not to be confused with scissor rect
	bool fullscreen; ///< Flag indicating whether or not the applicaion is in fullscreen.

	// GLFW callbacks
	friend void framebufferSizeCallback(GLFWwindow *window, int width, int height);
	friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	friend void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
	friend void windowPosCallback(GLFWwindow *window, int xpos, int ypos);
	friend void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
	friend void windowCloseCallback(GLFWwindow *window);
};

App *getApp();

}
