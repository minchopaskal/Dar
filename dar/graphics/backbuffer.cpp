#include "backbuffer.h"

#include "framework/app.h"

namespace Dar {

bool Backbuffer::init(ComPtr<IDXGIFactory4> dxgiFactory, CommandQueue &commandQueue, bool allowTearing) {
	App *app = getApp();

	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	scDesc.Width = app->getWidth();
	scDesc.Height = app->getHeight();
	scDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	scDesc.Stereo = FALSE;
	// NOTE: if multisampling should be enabled the bitblt transfer swap method should be used
	scDesc.SampleDesc = { 1, 0 }; // Not using multi-sampling.
	scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scDesc.BufferCount = FRAME_COUNT;
	scDesc.Scaling = DXGI_SCALING_STRETCH;
	scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
	scDesc.Flags = allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

	ComPtr<IDXGISwapChain1> swapChainPlaceholder;
	RETURN_FALSE_ON_ERROR(
		dxgiFactory->CreateSwapChainForHwnd(
			commandQueue.getCommandQueue().Get(), // Swap chain needs the queue so that it can force a flush on it.
			app->getWindow(),
			&scDesc,
			NULL /*DXGI_SWAP_CHAIN_FULLSCREEN_DESC*/, // will be handled manually
			NULL /*pRestrictToOutput*/,
			&swapChainPlaceholder
		),
		"Failed to create swap chain!"
	);

	RETURN_FALSE_ON_ERROR(
		dxgiFactory->MakeWindowAssociation(app->getWindow(), DXGI_MWA_NO_ALT_ENTER),
		"Failed to make window association NO_ALT_ENTER"
	);

	RETURN_FALSE_ON_ERROR(
		swapChainPlaceholder.As(&swapChain),
		"Failed to create swap chain!"
	);

	for (UINT i = 0; i < FRAME_COUNT; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);

		wchar_t backBufferName[32];
		swprintf(backBufferName, 32, L"BackBuffer[%u]", i);
		backBuffers[i]->SetName(backBufferName);
	}

	return true;
}

bool Backbuffer::registerInResourceManager() {
	auto &resManager = getResourceManager();

	for (UINT i = 0; i < FRAME_COUNT; ++i) {
		RETURN_FALSE_ON_ERROR_FMT(
			swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])),
			"Failed to create Render-Target-View for buffer %u!", i
		);

		// Register the back buffer's resources manually since the resource manager doesn't own them, the swap chain does.
#ifdef DAR_DEBUG
		backBuffersHandles[i] = resManager.registerResource(
			backBuffers[i].Get(),
			1, /*numSubresource*/
			0, /*size*/ // we don't use that
			D3D12_RESOURCE_STATE_COMMON, // initial state
			ResourceType::RenderTargetBuffer
		);
#else
		backBuffersHandles[i] = resManager.registerResource(backBuffers[i].Get(), 1, 0, D3D12_RESOURCE_STATE_COMMON);
#endif

		wchar_t backBufferName[32];
		swprintf(backBufferName, 32, L"BackBuffer[%u]", i);
		backBuffers[i]->SetName(backBufferName);
	}

	return true;
}

void Backbuffer::registerRenderer(Renderer * r) {
	renderers.push_back(r);
}

bool Backbuffer::resize() {
	auto &resManager = getResourceManager();

	App *app = getApp();
	int width = app->getWidth();
	int height = app->getHeight();

	for (unsigned int i = 0; i < FRAME_COUNT; ++i) {
		backBuffers[i].Reset();
		// It's important to deregister an outside resource if you want it deallocated
		// since the ResourceManager keeps a ref if it was registered with it.
		resManager.deregisterResource(backBuffersHandles[i]);
	}

	DXGI_SWAP_CHAIN_DESC scDesc = { };
	RETURN_FALSE_ON_ERROR(
		swapChain->GetDesc(&scDesc),
		"Failed to retrieve swap chain's description"
	);
	RETURN_FALSE_ON_ERROR(
		swapChain->ResizeBuffers(
			FRAME_COUNT,
			width,
			height,
			scDesc.BufferDesc.Format,
			scDesc.Flags
		),
		"Failed to resize swap chain buffer"
	);

	RETURN_FALSE_ON_ERROR(registerInResourceManager(), "Failed to register resized backbuffer in resource manager!");

	for (auto r : renderers) {
		if (r != nullptr) {
			r->onBackbufferResize();
		}
	}

	return true;
}

UINT Backbuffer::getCurrentBackBufferIndex() const {
	return swapChain->GetCurrentBackBufferIndex();
}

DXGI_FORMAT Backbuffer::getFormat() const {
	DXGI_SWAP_CHAIN_DESC1 scDesc = {};
	RETURN_ON_ERROR(swapChain->GetDesc1(&scDesc), DXGI_FORMAT_UNKNOWN, "Failed to retrieve swap chain's description");

	return scDesc.Format;
}

HRESULT Backbuffer::present(UINT syncInterval, UINT flags) const {
	return swapChain->Present(syncInterval, flags);
}

void Backbuffer::deinit() {
	swapChain.Reset();
	for (int i = 0; i < FRAME_COUNT; ++i) {
		backBuffers[i].Reset();
	}
}

} // namespace Dar
