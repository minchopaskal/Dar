#pragma once 

#include "graphics/d3d12/data_buffer.h"

namespace Dar {

enum class ReadbackBufferMapError {
	Success = 0,

	NoReadbackbuffer,
	AlreadyMapped,
	D3D12Error,
};

class ReadWriteBufferResource {
public:
	/// Intialize or resize the data buffer.
	/// @param elemSize Size of the elements in the buffer to be allocated.
	/// @param numElems Number of elements in the buffer to be allocated.
	/// @param readback Wheather this buffer will be readback on CPU.
	/// @param heapInfo Optional pointer to a HeapInfo object describing the heap on which we want the allocation to happen.
	/// @return true on success, false otherwise.
	bool init(SizeType elemSize, SizeType numElems, bool readback, Optional<HeapHandle> heapHandle = std::nullopt);

	/// Upload CPU data to the buffer.
	/// @param uploadHandle Upload handle object created by the resource manager. Used when batch uploading.
	/// @param data Data to be uploaded.
	/// @return true on success, false otherwise.
	bool upload(UploadHandle uploadHandle, const void *data);

	ResourceHandle getHandle() const {
		return bufferHandle;
	}

	ResourceHandle getReeadbackBufferHandle() const {
		return readbackBufferHandle;
	}

	bool hasReadbackBuffer() const {
		return getReeadbackBufferHandle() != INVALID_RESOURCE_HANDLE;
	}

	/// @brief Map the readback buffer to a cpu buffer.
	/// @param cpuBuffer Pointer to the pointer of the cpu buffer we wish to map the resource to.
	/// @return false if there is no readback buffer or mapping error occurs.
	ReadbackBufferMapError mapReadbackBuffer(void **cpuBuffer);

	/// @brief Unmaps the readback buffer. One must unmap the readback buffer after each mapping
	/// or if one wishes to map it to another buffer.
	void unmapReadbackBuffer();

	SizeType getSize() const {
		return size;
	}

	SizeType getNumElements() const {
		return size / elementSize;
	}

	SizeType getElementSize() const {
		return elementSize;
	}

	void setName(const String &name) {
		bufferHandle.get()->SetName(strToWStr(name).c_str());
	}

	void deinit();

	DataBufferResource asDataBuffer() const;

private:
	ResourceHandle bufferHandle = INVALID_RESOURCE_HANDLE;
	ResourceHandle readbackBufferHandle = INVALID_RESOURCE_HANDLE;
	SizeType size = 0;
	SizeType elementSize = 0;
	bool mapped = false;
};

} // namespace Dar
