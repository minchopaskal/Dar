#include "d3d12_utils.h"

#include <cassert>

int updateResource(
	ComPtr<ID3D12Device2> &device,
	ComPtr<ID3D12GraphicsCommandList2> &commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	D3D12_SUBRESOURCE_DATA *subres,
	D3D12_RESOURCE_DESC desc,
	D3D12_RESOURCE_FLAGS flags
) {
	assert(stagingBuffer != nullptr);

	CD3DX12_HEAP_PROPERTIES dstProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	D3D12_RESOURCE_STATES dstState = D3D12_RESOURCE_STATE_COPY_DEST;

	RETURN_FALSE_ON_ERROR(
		device->CreateCommittedResource(
			&dstProps,
			D3D12_HEAP_FLAG_NONE,
			&desc,
			dstState,
			nullptr,
			IID_PPV_ARGS(destinationResource)
		),
		"Failed to create GPU resource!\n"
	);

	auto size = GetRequiredIntermediateSize(*destinationResource, 0, 1);

	RETURN_FALSE_ON_ERROR(
		device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(size),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(stagingBuffer)
		),
		"Failed to create staging buffer resource!\n"
	);

	assert(subres);

	return UpdateSubresources(commandList.Get(), *destinationResource, *stagingBuffer, 0, 0, 1, subres);
}

int updateBufferResource(
	ComPtr<ID3D12Device2> &device,
	ComPtr<ID3D12GraphicsCommandList2> &commandList,
	ID3D12Resource **destinationResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	D3D12_RESOURCE_FLAGS flags
) {
	auto desc = CD3DX12_RESOURCE_DESC::Buffer(cpuBuffer.size, flags);
	D3D12_SUBRESOURCE_DATA subres = {};
	subres.pData = cpuBuffer.ptr;
	subres.RowPitch = cpuBuffer.size;
	subres.SlicePitch = 1;
	return updateResource(
		device,
		commandList,
		destinationResource,
		stagingBuffer,
		&subres,
		desc,
		flags
	);
}

int updateTex2DResource(
	ComPtr<ID3D12Device2> &device,
	ComPtr<ID3D12GraphicsCommandList2> &commandList,
	ID3D12Resource **textureResource,
	ID3D12Resource **stagingBuffer,
	CPUBuffer cpuBuffer,
	UINT width,
	UINT height,
	DXGI_FORMAT format,
	D3D12_RESOURCE_FLAGS flags
) {
	assert(stagingBuffer != nullptr);

	D3D12_RESOURCE_DESC textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1, 1);
	D3D12_SUBRESOURCE_DATA subres = {};
	subres.pData = cpuBuffer.ptr;
	subres.RowPitch = width * 4;
	subres.SlicePitch = subres.RowPitch * height;

	if (!updateResource(
		device,
		commandList,
		textureResource,
		stagingBuffer,
		&subres,
		textureDesc,
		flags
	)) {
		return false;
	}

	return true;
}

WString getPrefixedNameByType(D3D12_COMMAND_LIST_TYPE type, LPWSTR prefix) {
	WString prefixStr{ prefix };
	switch (type) {
	case D3D12_COMMAND_LIST_TYPE_DIRECT:
		prefixStr.append(L"Direct").c_str();
		break;
	case D3D12_COMMAND_LIST_TYPE_COPY:
		prefixStr.append(L"Copy");
		break;
	case D3D12_COMMAND_LIST_TYPE_COMPUTE:
		prefixStr.append(L"Compute");
		break;
	default:
		prefixStr.append(L"Generic");
		break;
	}

	return prefixStr;
}

WString getCommandQueueNameByType(D3D12_COMMAND_LIST_TYPE type) {
	return getPrefixedNameByType(type, L"CommandQueue");
}

WString getCommandListNameByType(D3D12_COMMAND_LIST_TYPE type) {
	return getPrefixedNameByType(type, L"CommandList");
}
