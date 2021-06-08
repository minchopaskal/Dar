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
	ComPtr<ID3D12Device2> device,
	ComPtr<ID3D12GraphicsCommandList2> commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	D3D12_RESOURCE_FLAGS flags,
	D3D12_RESOURCE_STATES dstDesiredState
);
