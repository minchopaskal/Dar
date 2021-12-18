#pragma once

#include "d3d12_includes.h"

#include "d3d12_command_queue.h"

#include <bitset>

struct GLFWwindow;
struct ResourceManager;
struct CommandList;

struct D3D12App {
	D3D12App(UINT width, UINT height, const char *windowTitle);
	virtual ~D3D12App();

	/// The main loop.
	int run();

	/// Toggle between windowed/fullscreen
	void toggleFullscreen();

	/// Get pointer to the window.
	HWND getWindow() const;

	/// Flush any command queues' work.
	/// NB doesn't flush the resource manager's copy Queue.
	void flush();

	/// Initialization work. Overrieds should at leat call D3D12App::init() in order to
	/// initialize DirectX12 and other systems.
	virtual int init() = 0;

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

	/// Optional. Should be called during init of the derived class if one intends to draw UI via ImGui.
	/// Note: call before D3D12::init() in order to skip ImGui initialization.
	void setUseImGui();

	/// Optional. It is advised that all ImGui draw calls go here, unless it's inconvinient.
	/// Called during renderUI().
	virtual void drawUI() { };

	/// Optional. Should be called before the last transition of the RTV to PRESENT state
	/// if the app wants its ImGui draw calls rendered.
	void renderUI(CommandList &cmdList, D3D12_CPU_DESCRIPTOR_HANDLE &rtvHandle);

	/// Return width of window
	int getWidth() const;

	/// Return height of window
	int getHeight() const;

	// TODO: mouse callback, etc.
	virtual void onResize(const unsigned int w, const unsigned int h) = 0;
	virtual void onKeyboardInput(int key, int action) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset) = 0;
	virtual void onWindowPosChange(int xPos, int yPos) { }

public:
	bool vSyncEnabled;
	bool allowTearing;

protected:
	static const UINT frameCount = 2;
	
	// TODO: get this out of here
	// Input
	static const int keysCount = 90; // see GLFW_KEY_Z
	Bitset<keysCount> keyPressed;
	Bitset<keysCount> keyRepeated;

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

	char title[256]; ///< Title of the window
	UINT width, height; ///< Dimensions of the window

	int abort; ///< Abort the main loop if true.

private:
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
	friend void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	friend void windowPosCallback(GLFWwindow *window, int xpos, int ypos);
};
