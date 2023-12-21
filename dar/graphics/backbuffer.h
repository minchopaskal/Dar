#pragma once

#include "core.h"
#include "d3d12/command_queue.h"
#include "utils/defines.h"

namespace Dar {

class Renderer;

struct Backbuffer {
	bool init(ComPtr<IDXGIFactory4> dxgiFactory, CommandQueue &commandQueue, bool allowTearing);

	ID3D12Resource *getBufferResource(UINT backbufferIndex) const {
		return backBuffers[backbufferIndex].Get();
	}

	ResourceHandle getHandle(UINT backbufferIndex) const {
		return backBuffersHandles[backbufferIndex];
	}

	bool registerInResourceManager();

	void registerRenderer(Renderer *r);

	bool resize();

	UINT getCurrentBackBufferIndex() const;

	DXGI_FORMAT getFormat() const;

	HRESULT present(UINT syncInterval, UINT flags) const;

	void deinit();

public:
	// Swap chain resources
	ComPtr<IDXGISwapChain4> swapChain; ///< Pointer to the swap chain
	ComPtr<ID3D12Resource> backBuffers[FRAME_COUNT]; ///< The RT resources
	ResourceHandle backBuffersHandles[FRAME_COUNT]; ///< Handles to the RT resources in the resource manager

	Vector<Renderer*> renderers; ///< Renderers using the backbuffer
};

} // namespace Dar
