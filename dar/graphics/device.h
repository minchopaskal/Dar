#pragma once

#include "backbuffer.h"
#include "d3d12/includes.h"

namespace Dar {

class Device {
public:
	Device();

	bool init();
	void deinit();

	ComPtr<ID3D12Device> getDevice() const;
	ID3D12Device* getDevicePtr() const;

	void registerBackBuffersInResourceManager();
	bool resizeBackBuffers();

	void flushCommandQueue() {
		commandQueueDirect.flush();
	}

	CommandList getCommandList() {
		return commandQueueDirect.getCommandList();
	}

	CommandQueue& getCommandQueue() {
		return commandQueueDirect;
	}

	Backbuffer& getBackbuffer() {
		return backbuffer;
	}

	bool getAllowTearing() const {
		return allowTearing;
	}

	bool createSRVHeap(D3D12_DESCRIPTOR_HEAP_DESC desc, ComPtr<ID3D12DescriptorHeap> &heap) {
		RETURN_FALSE_ON_ERROR(
			device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)),
			"Failed to create ImGui's SRV descriptor heap!"
		);

		return true;
	}

	ComPtr<ID3D12DescriptorHeap> getImGuiSRVHeap() const {
		return imguiSRVHeap.Get();
	}

private:
	bool initImGui();
	bool deinitImGui();

private:
	ComPtr<ID3D12Device> device;

	/// General command queue. Primarily used for draw calls.
	/// Copy calls are handled by the resource manager
	CommandQueue commandQueueDirect;

	Backbuffer backbuffer; ///< Object holding the data for the backbuffers.

	// Imgui
	ComPtr<ID3D12DescriptorHeap> imguiSRVHeap; ///< SRVHeap used by Dear ImGui for font drawing
	
	/// Flag that indicates ImGui was already shutdown. Since ImGui doesn't do check for double delete
	/// in its DX12 implementation of Shutdown, we have to do it manually.
	bool imGuiShutdown = false;

	bool allowTearing = false;
};

}