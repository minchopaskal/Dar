#pragma once

#include "d3dx12.h"
#include "d3d12_defines.h"
#include "d3d12_includes.h"

#include <type_traits>

struct CPUBuffer {
	void *ptr;
	UINT64 size;
};

int updateBufferResource(
	ComPtr<ID3D12Device8> &device,
	ComPtr<ID3D12GraphicsCommandList2> &commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	D3D12_RESOURCE_FLAGS flags
);

int updateTex2DResource(
	ComPtr<ID3D12Device8> &device,
	ComPtr<ID3D12GraphicsCommandList2> &commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	UINT width,
	UINT height,
	DXGI_FORMAT format,
	D3D12_RESOURCE_FLAGS flags
);

WString getCommandQueueNameByType(D3D12_COMMAND_LIST_TYPE type);
WString getCommandListNameByType(D3D12_COMMAND_LIST_TYPE type);
