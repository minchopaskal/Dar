#pragma once

#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>

#include <D3d12.h>
#include <dxgi.h>
#include <dxgi1_5.h>

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#include "command_queue.h"

struct D3D12App {
	D3D12App(UINT width, UINT height, const char *windowTitle);
	virtual ~D3D12App();

	int run();
	void toggleFullscreen();
	HWND getWindow() const;

	virtual int init();
	virtual int loadAssets() = 0;
	virtual void deinit() = 0;
	virtual void update() = 0;
	virtual void render() = 0;

	// TODO: mouse callback, etc.
	virtual void resize(int width, int height) = 0;
	virtual void keyboardInput(int key, int action) = 0;

protected:
	bool updateRenderTargetViews();

public:
	bool vSyncEnabled;
	bool tearingEnabled;

protected:
	static const UINT frameCount = 3;

	ComPtr<ID3D12Device2> device;

	CommandQueue commandQueueDirect;

	ComPtr<IDXGISwapChain4> swapChain;
	ComPtr<ID3D12DescriptorHeap> rtvHeap;
	ComPtr<ID3D12Resource> backBuffers[frameCount];
	UINT64 fenceValues[frameCount];

	UINT frameIndex;
	UINT rtvHeapHandleIncrementSize;

	char title[256];
	UINT width, height;

private:
	HWND window;
	RECT windowRect;
	bool fullscreen;
};
