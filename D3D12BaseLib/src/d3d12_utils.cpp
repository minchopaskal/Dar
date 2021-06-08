#include "d3d12_utils.h"

int updateBufferResource(
	ComPtr<ID3D12Device2> device,
	ComPtr<ID3D12GraphicsCommandList2> commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	D3D12_RESOURCE_FLAGS flags,
	D3D12_RESOURCE_STATES dstDesiredState
) {
	if (cpuBuffer.size < 1 || cpuBuffer.ptr == nullptr) {
		return false;
	}

	const bool useStagingBuffer = (stagingBuffer != nullptr);

	CD3DX12_HEAP_PROPERTIES dstProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	D3D12_RESOURCE_STATES dstState = D3D12_RESOURCE_STATE_COPY_DEST;
	if (!useStagingBuffer) {
		dstProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		dstState = D3D12_RESOURCE_STATE_GENERIC_READ;
	}

	RETURN_FALSE_ON_ERROR(
		device->CreateCommittedResource(
		&dstProps,
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(cpuBuffer.size, flags),
		dstState,
		nullptr,
		IID_PPV_ARGS(destinationResource)
	),
		"Failed to create GPU resource!\n"
	);

	// Create staging buffer if needed
	ID3D12Resource *copyToBuffer = *destinationResource;
	if (useStagingBuffer) {
		RETURN_FALSE_ON_ERROR(
			device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(cpuBuffer.size),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(stagingBuffer)
		),
			"Failed to create staging buffer resource!\n"
		);
		copyToBuffer = *stagingBuffer;
	}

	// Copy CPU data to the staging buffer
	UINT8 *stagingData = nullptr;
	D3D12_RANGE range = { 0, 0 };
	RETURN_FALSE_ON_ERROR(
		copyToBuffer->Map(0, &range, reinterpret_cast<void**>(&stagingData)),
		"Couldn't map staging buffer to CPU data!\n"
	);

	memcpy(stagingData, cpuBuffer.ptr, cpuBuffer.size);
	copyToBuffer->Unmap(0, nullptr);


	if (useStagingBuffer) {
		commandList->CopyResource(*destinationResource, *stagingBuffer);
	}

	return true;
}