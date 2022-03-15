#pragma once

#include "d3d12_descriptor_heap.h"
#include "d3d12_includes.h"
#include "d3d12_resource_handle.h"

struct DepthBuffer {
	/// Intialize or resize the depth buffer.
	/// @param device Device used for creation of the DSV heap.
	/// @param width Width of the depth buffer texture
	/// @param height Height of the depth buffer texture
	/// @param format Format of the depth buffer. Must be one of DXGI_FORMAT_D* types.
	/// @return true on success, false otherwise
	bool init(ComPtr<ID3D12Device> device, int width, int height, DXGI_FORMAT format);

	/// @return Format of the depth buffer when used as a depth attachment.
	DXGI_FORMAT getFormatAsDepthBuffer() const;

	/// @return Format of the depth buffer when used  as a texture.
	DXGI_FORMAT getFormatAsTexture() const;

	D3D12_CPU_DESCRIPTOR_HANDLE getCPUHandle() const;

	ID3D12Resource *getBufferResource();

	ResourceHandle getBufferHandle() const;

private:
	DescriptorHeap dsvHeap;
	ResourceHandle bufferHandle = INVALID_RESOURCE_HANDLE;
	DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
	int width = 0;
	int height = 0;
};