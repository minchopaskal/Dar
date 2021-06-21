#pragma once

#include "d3d12_includes.h"

#include "d3d12_command_queue.h"

#include <bitset>

struct GLFWwindow;

struct D3D12App {
	D3D12App(UINT width, UINT height, const char *windowTitle);
	virtual ~D3D12App();

	int run();
	void toggleFullscreen();
	HWND getWindow() const;
	void flush();

	virtual int init();
	virtual int loadAssets() = 0;
	virtual void deinit() = 0;
	virtual void update() = 0;
	virtual void render() = 0;

	// TODO: mouse callback, etc.
	virtual void onResize(int width, int height) = 0;
	virtual void onKeyboardInput(int key, int action) = 0;
	virtual void onMouseScroll(double xOffset, double yOffset) = 0;
	virtual void onWindowPosChange(int xPos, int yPos) { }

public:
	bool vSyncEnabled;
	bool allowTearing;

protected:
	static const UINT frameCount = 2;
	
	// Input
	static const int keysCount = 90; // see GLFW_KEY_Z
	Bitset<keysCount> keyPressed;
	Bitset<keysCount> keyRepeated;

	ComPtr<ID3D12Device8> device;

	CommandQueue commandQueueDirect;
	CommandQueue commandQueueCopy;

	ComPtr<IDXGISwapChain4> swapChain;
	ComPtr<ID3D12Resource> backBuffers[frameCount];

	D3D12_FEATURE_DATA_ROOT_SIGNATURE rootSignatureFeatureData;

	UINT frameIndex;

	char title[256];
	UINT width, height;

private:
	HWND window;
	RECT windowRect;
	bool fullscreen;

	// GLFW callbacks
	friend void framebufferSizeCallback(GLFWwindow *window, int width, int height);
	friend void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
	friend void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	friend void windowPosCallback(GLFWwindow *window, int xpos, int ypos);
};
