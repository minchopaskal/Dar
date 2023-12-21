#pragma once

#include "d3d12/includes.h"
#include "d3d12/texture_res.h"
#include "utils/defines.h"

namespace Dar {

struct DescriptorHeap {
	DescriptorHeap();

	void init(
		ID3D12Device *device,
		D3D12_DESCRIPTOR_HEAP_TYPE type,
		UINT numDesctiptors,
		bool shaderVisible
	);

	/// Reset the heap so we can create new views starting from the beginning. 
	void reset();

	/// Add a texture2D SRV at the end of the heap.
	void addTexture2DSRV(const TextureResource &tex);

	/// Add a texture2D SRV at the end of the heap.
	void addTextureCubeSRV(const TextureResource &tex);

	/// Add a buffer SRV at the end of the heap.
	void addBufferSRV(ID3D12Resource *resource, UINT numElements, UINT elementSize);

	void addRTV(ID3D12Resource *resource, D3D12_RENDER_TARGET_VIEW_DESC *rtvDesc);

	void addDSV(ID3D12Resource *resource, DXGI_FORMAT foramt);

	ID3D12DescriptorHeap *const *getAddressOf() const {
		return heap.GetAddressOf();
	}

	D3D12_CPU_DESCRIPTOR_HANDLE getCPUHandle(int index) const {
		D3D12_CPU_DESCRIPTOR_HANDLE res = cpuHandleStart;
		res.ptr += static_cast<SIZE_T>(index) * handleIncrementSize;
		return res;
	}

	D3D12_GPU_DESCRIPTOR_HANDLE getGPUHandle(int index) const {
		D3D12_GPU_DESCRIPTOR_HANDLE res = gpuHandleStart;
		res.ptr += index * handleIncrementSize;
		return res;
	}

	operator bool() const {
		return !!heap;
	}

	SizeType getNumViews() const {
		auto desc = heap->GetDesc();
		return desc.NumDescriptors;
	}

	void deinit();

private:
	ComPtr<ID3D12DescriptorHeap> heap;
	ID3D12Device *device; /// Non-owning pointer to the device.

	D3D12_DESCRIPTOR_HEAP_TYPE type;

	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandleStart;
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandleStart;
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandleRunning;

	UINT handleIncrementSize;
};

} // namespace Dar