#include "read_write_buffer.h"

namespace Dar {

bool ReadWriteBufferResource::init(SizeType elemSize, SizeType numElems, bool readback, Optional<HeapHandle> heapHandle) {
	ResourceInitData initData(ResourceType::ReadWriteBuffer);
	initData.size = elemSize * numElems;
	initData.heapHandle = heapHandle;

	if (elementSize == elemSize && initData.size == size && bufferHandle != INVALID_RESOURCE_HANDLE) {
		return true;
	}

	deinit();

	auto &resManager = getResourceManager();
	bufferHandle = resManager.createBuffer(initData);

	if (bufferHandle == INVALID_RESOURCE_HANDLE) {
		return false;
	}

	if (readback) {
		ResourceInitData readbackInitData(ResourceType::ReadbackBuffer);
		readbackInitData.state = D3D12_RESOURCE_STATE_COPY_DEST;
		readbackInitData.size = elemSize * numElems;

		readbackBufferHandle = resManager.createBuffer(readbackInitData);
		if (readbackBufferHandle == INVALID_RESOURCE_HANDLE) {
			return false;
		}
	}

	size = initData.size;
	elementSize = elemSize;

	return true;
}

bool ReadWriteBufferResource::upload(UploadHandle uploadHandle, const void *data) {
	auto &resManager = getResourceManager();
	return resManager.uploadBufferData(uploadHandle, bufferHandle, data, size);
}

ReadbackBufferMapError ReadWriteBufferResource::mapReadbackBuffer(void **cpuBuffer) {
	if (mapped) {
		return ReadbackBufferMapError::AlreadyMapped;
	}

	if (readbackBufferHandle == INVALID_RESOURCE_HANDLE) {
		return ReadbackBufferMapError::NoReadbackbuffer;
	}

	D3D12_RANGE range = { 0, size };
	RETURN_ON_ERROR(readbackBufferHandle.get()->Map(0, &range, cpuBuffer), ReadbackBufferMapError::D3D12Error, "Failed to map readback buffer!");

	mapped = true;
	return ReadbackBufferMapError::Success;
}

void ReadWriteBufferResource::unmapReadbackBuffer() {
	if (!mapped || readbackBufferHandle == INVALID_RESOURCE_HANDLE) {
		return;
	}

	D3D12_RANGE emptyRange = { 0, 0 };
	readbackBufferHandle.get()->Unmap(0, &emptyRange);
}

void ReadWriteBufferResource::deinit() {
	unmapReadbackBuffer();

	if (bufferHandle != INVALID_RESOURCE_HANDLE) {
		auto &resManager = getResourceManager();
		resManager.deregisterResource(bufferHandle);

		if (readbackBufferHandle != INVALID_RESOURCE_HANDLE) {
			resManager.deregisterResource(readbackBufferHandle);
		}
	}

	size = 0;
	elementSize = 0;
	mapped = false;
}

DataBufferResource ReadWriteBufferResource::asDataBuffer() const {
	DataBufferResource res;
	res.handle = bufferHandle;
	res.size = size;
	res.elementSize = elementSize;

	return res;
}

} // namespace Dar
