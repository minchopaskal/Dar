#include "d3d12/depth_buffer.h"
#include "d3d12/resource_manager.h"
#include "data_buffer.h"

namespace Dar {

bool DataBufferResource::init(SizeType elemSize, SizeType numElems, Optional<HeapHandle> heapHandle) {
	ResourceInitData initData(ResourceType::DataBuffer);
	initData.size = elemSize * numElems;
	initData.heapHandle = heapHandle;
	initData.state = D3D12_RESOURCE_STATE_COMMON;

	if (elementSize == elemSize && initData.size == size && initData.heapHandle == heapHandle && handle != INVALID_RESOURCE_HANDLE) {
		return true;
	}

	deinit();

	auto &resManager = getResourceManager();
	handle = resManager.createBuffer(initData);

	if (handle == INVALID_RESOURCE_HANDLE) {
		return false;
	}

	size = initData.size;
	elementSize = elemSize;

	return true;
}

bool DataBufferResource::upload(UploadHandle uploadHandle, const void *data) {
	auto &resManager = getResourceManager();
	return resManager.uploadBufferData(uploadHandle, handle, data, size);
}

void DataBufferResource::deinit() {
	if (handle != INVALID_RESOURCE_HANDLE) {
		auto &resManager = getResourceManager();
		resManager.deregisterResource(handle);
	}

	size = 0;
	elementSize = 0;
}


} // namespace Dar