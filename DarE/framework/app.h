#pragma once

#include "d3d12/includes.h"

#include "d3d12/command_queue.h"

#include "framework/input_query.h"

struct GLFWwindow;
struct ResourceManager;
struct CommandList;
namespace Dar {
namespace JobSystem {
struct Fence;
}
}

struct D3D12App : public IKeyboardInputQuery {
	D3D12App(UINT width, UINT height, const char *windowTitle);
	virtual ~D3D12App();

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

protected:
	/// Toggle between windowed/fullscreen
	void toggleFullscreen();

	/// Get pointer to the window.
	HWND getWindow() const;

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

	/// Any render work should go here.
	virtual void render() = 0;

	/// Any initialization work should go here. Called inside init().
	/// @return 0 on failure, 1 on success
	virtual int initImpl() = 0;

	/// Optional. Should be called during init of the derived class if one intends to draw UI via ImGui.
	/// Note: call before D3D12::init() in order to skip ImGui initialization.
	void setUseImGui();

	/// Optional. It is advised that all ImGui draw calls go here, unless it's inconvinient.
	/// Called during renderUI().
	virtual void drawUI() {};

	/// Optional. Should be called before the last transition of the RTV to PRESENT state
	/// if the app wants its ImGui draw calls rendered.
	void renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle);

	GLFWwindow *getGLFWWindow() const;

	virtual void onResize(const unsigned int w, const unsigned int h) = 0;
	virtual void onKeyboardInput(int key, int action) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset) = 0;
	virtual void onMouseMove(double xOffset, double yOffset) {};
	virtual void onWindowPosChange(int xPos, int yPos) {}
	virtual void onWindowClose() {}

private:
	int initJob();
	int mainLoopJob();

protected:
	static const UINT frameCount = 2;

	// TODO: get this out of here
	// Input
	static const int keysCount = GLFW_KEY_LAST;
	Bitset<keysCount> keyPressed;
	Bitset<keysCount> keyRepeated;
	Bitset<keysCount> keyReleased;

	// TODO: maybe abstract it or somehow make it global.
	ComPtr<ID3D12Device8> device; ///< DX12 device used across all the classes

	/// General command queue. Primarily used for draw calls.
	/// Copy calls are handled by the resource manager
	CommandQueue commandQueueDirect;

	ComPtr<IDXGISwapChain4> swapChain; ///< Pointer to the swap chain
	ComPtr<ID3D12Resource> backBuffers[frameCount]; ///< The RT resources
	ResourceHandle backBuffersHandles[frameCount]; ///< Handles to the RT resources in the resource manager

	ResourceManager *resManager; ///< Pointer to the resource manager singleton

	D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignatureFeatureData; ///< Cache to the feature level for the root signature. Used when creating pipelines.

	ComPtr<ID3D12DescriptorHeap> imguiSRVHeap; ///< SRVHeap used by Dear ImGui for font drawing

	UINT frameIndex; ///< Current backbuffer index

	UINT numRenderedFrames = 0;

	char title[256]; ///< Title of the window
	UINT width, height; ///< Dimensions of the window

	Atomic<int> abort; ///< Abort the main loop if true.

	bool vSyncEnabled;
	bool allowTearing;

private:
	Dar::JobSystem::Fence *initJobFence = nullptr;

	HWND window; ///< Pointer to the win32 window abstraction
	RECT windowRect; ///< Window rectangle. Not to be confused with scissor rect
	bool fullscreen; ///< Flag indicating whether or not the applicaion is in fullscreen.
	bool useImGui; ///< Flag indicating whether ImGui will be used for drawing UI.
	/// Flag that indicates ImGui was already shutdown. Since ImGui doesn't do check for double delete
	/// in its DX12 implementation of Shutdown, we have to do it manually.
	bool imGuiShutdown;

	// GLFW callbacks
	friend void framebufferSizeCallback(GLFWwindow *window, int width, int height);
	friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	friend void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
	friend void windowPosCallback(GLFWwindow *window, int xpos, int ypos);
	friend void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
	friend void windowCloseCallback(GLFWwindow *window);
};
